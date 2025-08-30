import os
import time
import random
from typing import Callable, Iterable, Optional, Tuple, Type, Any

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


DEFAULT_ATTEMPTS = _env_int("TB_RETRY_ATTEMPTS", 5)
BASE_DELAY = _env_float("TB_RETRY_BASE_DELAY", 0.5)
MAX_DELAY = _env_float("TB_RETRY_MAX_DELAY", 8.0)
JITTER = _env_float("TB_RETRY_JITTER", 1.0)
# CSV list of status codes
_STATUS_CSV = os.getenv("TB_RETRY_STATUS_CODES", "408,429,500,502,503,504")
RETRY_STATUS_CODES = {int(s.strip()) for s in _STATUS_CSV.split(",") if s.strip().isdigit()}


def _next_sleep(attempt: int) -> float:
    # Exponential with full jitter
    delay = min(MAX_DELAY, BASE_DELAY * (2 ** (attempt - 1)))
    if JITTER > 0:
        delay = max(0.0, delay + random.uniform(-JITTER / 2.0, JITTER / 2.0))
    return delay


def retry_call(
    fn: Callable[[], Any],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    retry_exceptions: Tuple[Type[BaseException], ...] = (
        TimeoutError,
        ConnectionError,
    ),
    retry_status_codes: Optional[Iterable[int]] = None,
    on_retry: Optional[Callable[[int, Optional[int], Exception, float], None]] = None,
) -> Any:
    """
    Execute fn() with retries on exceptions and optional HTTP status codes.

    - If fn() raises an exception in retry_exceptions, it will retry.
    - If fn() returns an httpx.Response-like object and status is in retry_status_codes,
      we raise a synthetic exception to trigger retry.
    - on_retry(attempt, status_or_none, exc, sleep_s) is called before sleeping.
    """
    if retry_status_codes is None:
        retry_status_codes = RETRY_STATUS_CODES

    last_exc: Optional[Exception] = None
    for i in range(1, max(1, attempts) + 1):
        try:
            result = fn()
            # If result looks like an httpx.Response, inspect status
            if httpx is not None and isinstance(result, getattr(httpx, "Response", object)):
                status = getattr(result, "status_code", None)
                if status in set(retry_status_codes or {}):
                    raise RuntimeError(f"Retryable status: {status}")
            return result
        except Exception as e:  # noqa: BLE001
            last_exc = e
            status_for_cb: Optional[int] = None
            # If it is an httpx.HTTPStatusError, extract code for callback
            if httpx is not None and isinstance(e, getattr(httpx, "HTTPStatusError", tuple())):
                status_for_cb = getattr(getattr(e, "response", None), "status_code", None)
            # If exception not in retryable set, re-raise immediately
            if not isinstance(e, retry_exceptions):
                # Allow retry on synthetic status-based RuntimeError
                if not (isinstance(e, RuntimeError) and "Retryable status:" in str(e)):
                    raise
            if i >= attempts:
                break
            sleep_s = _next_sleep(i)
            if on_retry is not None:
                try:
                    on_retry(i, status_for_cb, e, sleep_s)
                except Exception:
                    pass
            time.sleep(sleep_s)
    if last_exc is not None:
        raise last_exc
    # Fallback, should not happen
    raise RuntimeError("retry_call exhausted without exception")
