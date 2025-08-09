import os
import sys
import json
import socket
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv, find_dotenv


def _load_env() -> None:
    try:
        env_path = find_dotenv(usecwd=True)
        load_dotenv(env_path or None)
    except Exception:
        try:
            load_dotenv()
        except Exception:
            pass


def ensure_dirs() -> Tuple[bool, list[str]]:
    created: list[str] = []
    ok = True
    for d in ("runs", "bars"):
        p = Path(d)
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
                created.append(str(p))
            except Exception as e:
                print(f"[preflight] failed to create {d}: {e}")
                ok = False
    return ok, created


def _can_resolve(host: str) -> bool:
    try:
        socket.gethostbyname(host)
        return True
    except Exception:
        return False


def _check_telegram_reachability(timeout: float = 3.0) -> Tuple[bool, str]:
    """
    Basic network reachability check to Telegram API host. Does not send data.
    No secrets are required; uses DNS resolution and TCP connect.
    """
    host = "api.telegram.org"
    if not _can_resolve(host):
        return False, "DNS resolution failed for api.telegram.org"
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True, "ok"
    except Exception as e:
        return False, f"TCP connect failed: {e}"


def health(verbose: bool = False) -> int:
    _load_env()
    ok_dirs, created = ensure_dirs()

    tg_ok, tg_msg = _check_telegram_reachability()

    status = {
        "dirs_ok": ok_dirs,
        "dirs_created": created,
        "telegram_reachable": tg_ok,
        "telegram_note": tg_msg,
    }
    if verbose:
        print(json.dumps(status, indent=2))
    return 0 if (ok_dirs and tg_ok) else 1


if __name__ == "__main__":
    sys.exit(health(verbose=True))


