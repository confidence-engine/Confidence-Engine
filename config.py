import os
import unicodedata
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv

# Robust .env loading that works with python -c and other contexts
try:
    env_path = find_dotenv(usecwd=True)
    load_dotenv(env_path or None)
except Exception:
    try:
        load_dotenv()
    except Exception:
        pass

def _parse_bool(s: str, default: bool = False) -> bool:
    if s is None:
        return default
    return s.strip().lower() in ("1", "true", "yes", "on")

def _normalize_commas_and_spaces(s: str) -> str:
    if not s:
        return ""
    for ch in ["\uFF0C", "\uFE10", "\uFE50", "\u3001", "\u201A", "\u201E", "\u2E41"]:
        s = s.replace(ch, ",")
    s = "".join(" " if unicodedata.category(c).startswith("Z") else c for c in s)
    return s

def _parse_keys_list(env_var: str) -> list[str]:
    raw = os.getenv(env_var, "") or ""
    raw = _normalize_commas_and_spaces(raw).strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
            p = p[1:-1].strip()
        if p:
            out.append(p)
    return out

def _collect_numbered_keys(prefix: str = "PPLX_API_KEY_") -> list[str]:
    keys: list[str] = []
    i = 1
    while True:
        v = os.getenv(f"{prefix}{i}", "").strip()
        if not v:
            break
        keys.append(v)
        i += 1
    return keys

@dataclass
class Settings:
    """
    Settings loaded from environment with safe parsing and sensible defaults.

    Overrides precedence (highest first):
      1) Process environment variables (e.g., set by CLI wrapper `scripts/run.py`)
      2) Values from .env loaded via python-dotenv
      3) Hard defaults declared below

    Use TB_* envs for automation/testing controls (e.g., TB_NO_TELEGRAM) — they
    are read by the relevant modules (telegram_bot, scripts/run) rather than here.
    """
    alpaca_key_id: str = os.getenv("ALPACA_API_KEY_ID", "")
    alpaca_secret_key: str = os.getenv("ALPACA_API_SECRET_KEY", "")
    alpaca_base_url: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    symbol: str = os.getenv("SYMBOL", "BTC/USD")

    lookback_minutes: int = int(os.getenv("LOOKBACK_MINUTES", "120"))
    headlines_limit: int = int(os.getenv("HEADLINES_LIMIT", "10"))

    narrative_halflife_min: int = int(os.getenv("NARRATIVE_HALFLIFE_MIN", "90"))
    divergence_threshold: float = float(os.getenv("DIVERGENCE_THRESHOLD", "1.0"))
    confidence_cutoff: float = float(os.getenv("CONFIDENCE_CUTOFF", "0.6"))

    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.42"))
    use_coindesk: bool = _parse_bool(os.getenv("USE_COINDESK", "true"), default=True)

    pplx_enabled: bool = _parse_bool(os.getenv("PPLX_ENABLED", "true"), default=True)
    pplx_hours: int = int(os.getenv("PPLX_HOURS", "24"))
    pplx_api_keys: list[str] = None

    def __post_init__(self):
        # Normalize Alpaca base URL to avoid duplicated version path ("/v2/v2/")
        try:
            u = (self.alpaca_base_url or "").strip()
            if u:
                u = u.rstrip("/")
                if u.endswith("/v2"):
                    u = u[:-3]  # drop trailing '/v2'
            self.alpaca_base_url = u or "https://paper-api.alpaca.markets"
        except Exception:
            # Fallback to default if anything odd happens
            self.alpaca_base_url = "https://paper-api.alpaca.markets"
        # Priority 1: numbered keys PPLX_API_KEY_1, _2, ...
        numbered = _collect_numbered_keys("PPLX_API_KEY_")
        if numbered:
            self.pplx_api_keys = numbered
            return
        # Priority 2: comma-separated PPLX_API_KEYS
        list_keys = _parse_keys_list("PPLX_API_KEYS")
        if list_keys:
            self.pplx_api_keys = list_keys
            return
        # Priority 3: single PPLX_API_KEY
        single = (os.getenv("PPLX_API_KEY", "") or "").strip()
        self.pplx_api_keys = [single] if single else []

settings = Settings()
