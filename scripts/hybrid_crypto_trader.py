import os
import subprocess
import sys
import time
import logging
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime, timedelta, timezone
from pathlib import Path
import git
import numpy as np
import pandas as pd
import httpx
import torch

# Ensure project root on sys.path for local imports when executed directly
_THIS_DIR = Path(__file__).parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

try:
    from alpaca_trade_api.rest import REST  # type: ignore
except Exception:
    REST = None  # type: ignore

from config import settings
from scripts.discord_sender import send_discord_digest_to
from telegram_bot import send_message as send_telegram
from scripts.retry_utils import retry_call, RETRY_STATUS_CODES

# ==========================
# Logging
# ==========================
LOG_PATH = os.getenv("TB_TRADING_LOG", "trading_agent.log")
logger = logging.getLogger("hybrid_trader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)

# ==========================
# Env/Safety Flags
# ==========================
OFFLINE = os.getenv("TB_TRADER_OFFLINE", "1") in ("1", "true", "on", "yes")
NO_TRADE = os.getenv("TB_NO_TRADE", "1") in ("1", "true", "on", "yes")
NOTIFY = os.getenv("TB_TRADER_NOTIFY", "0") == "1"
ENABLE_DISCORD = os.getenv("TB_ENABLE_DISCORD", "0") == "1"
NO_TELEGRAM = os.getenv("TB_NO_TELEGRAM", "1").lower() in ("1", "true", "on", "yes")
HEARTBEAT = os.getenv("TB_TRADER_NOTIFY_HEARTBEAT", "0") == "1"
HEARTBEAT_EVERY_N = int(os.getenv("TB_HEARTBEAT_EVERY_N", "12"))  # every N runs

# Live auto-apply of promoted parameters (opt-in)
AUTO_APPLY_ENABLED = os.getenv("TB_AUTO_APPLY_ENABLED", "0").lower() in ("1", "true", "on", "yes")
AUTO_APPLY_KILL = os.getenv("TB_AUTO_APPLY_KILL", "0").lower() in ("1", "true", "on", "yes")

# ==========================
# Per-fill logging helpers
# ==========================
GATE_MODE = os.getenv("TB_GATE_MODE", "normal")

def log_order_event(event: str, symbol: str, side: str, qty, price=None, extra=None):
    """
    Structured log for stderr/file so it shows in trader_loop.err and trading_agent.log.
    event: 'order_submitted' | 'order_filled' | 'order_partially_filled'
    """
    try:
        q = float(qty) if qty is not None else None
    except Exception:
        q = qty
    payload: Dict[str, Any] = {
        "event": event,
        "mode": GATE_MODE,
        "symbol": symbol,
        "side": side,
        "qty": q,
    }
    if price is not None:
        try:
            payload["price"] = float(price)
        except Exception:
            payload["price"] = price
    if extra:
        try:
            payload.update(extra)
        except Exception:
            pass
    logger.info("[order] %s", payload)
    
def try_log_fill_once(api: Optional[REST], order_id: Optional[str], symbol: str, side: str) -> None:
    """Best-effort single status check to emit a fill log without blocking the loop.
    Controlled by TB_LOG_FILLS=1. Safe if REST or order_id is missing.
    """
    if os.getenv("TB_LOG_FILLS", "1") != "1":
        return
    if api is None or not order_id:
        return
    try:
        def _get():
            return api.get_order(order_id)
        def _on_retry(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
            # keep quiet; fills polling is best-effort
            return
        odr = retry_call(
            _get,
            attempts=1,
            retry_exceptions=(Exception,),
            retry_status_codes=RETRY_STATUS_CODES,
            on_retry=_on_retry,
        )
        status = str(getattr(odr, "status", "")).lower()
        filled_qty = getattr(odr, "filled_qty", None)
        filled_avg_price = getattr(odr, "filled_avg_price", None)
        if status == "filled":
            log_order_event(
                "order_filled",
                symbol=symbol,
                side=side,
                qty=filled_qty,
                price=filled_avg_price,
                extra={"order_id": order_id},
            )
        elif status in ("partially_filled", "partial"):  # tolerate variants
            log_order_event(
                "order_partially_filled",
                symbol=symbol,
                side=side,
                qty=filled_qty,
                price=filled_avg_price,
                extra={"order_id": order_id},
            )
    except Exception:
        # best-effort only
        return
PROMOTED_PARAMS_PATH = Path("config/promoted_params.json")
AUTO_APPLY_AUDIT_DIR = Path("eval_runs/live_auto_apply")

# Optional ML gate
USE_ML_GATE = os.getenv("TB_USE_ML_GATE", "0") == "1"
ML_MODEL_PATH = os.getenv("TB_ML_GATE_MODEL_PATH", "eval_runs/ml/latest/model.pt")
ML_FEATURES_PATH = os.getenv("TB_ML_FEATURES_PATH", "eval_runs/ml/latest/features.csv")
ML_MIN_PROB = float(os.getenv("TB_ML_GATE_MIN_PROB", "0.5"))
ML_SOFT_GATE = os.getenv("TB_ML_GATE_SOFT", "1") == "1"

# Signal debounce: require EMA12>EMA26 for last N bars (0 disables)
SIGNAL_DEBOUNCE_N = int(os.getenv("TB_SIGNAL_DEBOUNCE_N", "1"))

# Optional volatility and higher-timeframe regime filters
USE_ATR_FILTER = os.getenv("TB_USE_ATR_FILTER", "0") == "1"
ATR_LEN = int(os.getenv("TB_ATR_LEN", "14"))
ATR_MIN_PCT = float(os.getenv("TB_ATR_MIN_PCT", "0.0"))
ATR_MAX_PCT = float(os.getenv("TB_ATR_MAX_PCT", "1.0"))

USE_HTF_REGIME = os.getenv("TB_USE_HTF_REGIME", "0") == "1"
HTF_EMA_LEN = int(os.getenv("TB_HTF_EMA_LEN", "200"))  # 1h EMA200 as 4h regime proxy

# Optional 1h entry path (in addition to 15m). When enabled, a 1h EMA cross-up
# can trigger a smaller-sized entry even if 15m cross is quiet.
USE_1H_ENTRY = os.getenv("TB_USE_1H_ENTRY", "1") == "1"
EMA_1H_FAST = int(os.getenv("TB_1H_EMA_FAST", "12"))
EMA_1H_SLOW = int(os.getenv("TB_1H_EMA_SLOW", "26"))
DEBOUNCE_1H_N = int(os.getenv("TB_1H_DEBOUNCE_N", "1"))
SIZE_1H_MULT = float(os.getenv("TB_1H_SIZE_MULT", "0.5"))  # fraction of normal size

SYMBOL = os.getenv("SYMBOL", settings.symbol or "BTC/USD")
TF_FAST = "15Min"
TF_SLOW = "1Hour"

# Risk and brackets
MAX_PORTFOLIO_RISK = float(os.getenv("TB_MAX_RISK_FRAC", "0.01"))   # 1%
TP_PCT = float(os.getenv("TB_TP_PCT", "0.05"))                      # +5%
SL_PCT = float(os.getenv("TB_SL_PCT", "0.02"))                      # -2%
DAILY_LOSS_CAP_PCT = float(os.getenv("TB_DAILY_LOSS_CAP_PCT", "0.03"))  # 3% of reference equity

# Optional ATR-based stop sizing (replaces fixed SL_PCT when enabled)
USE_ATR_STOP = os.getenv("TB_USE_ATR_STOP", "0") == "1"
ATR_STOP_MULT = float(os.getenv("TB_ATR_STOP_MULT", "1.5"))

# Sentiment
SENTIMENT_THRESHOLD = float(os.getenv("TB_SENTIMENT_CUTOFF", "0.5"))
PPLX_TIMEOUT = float(os.getenv("TB_PPLX_TIMEOUT", "12"))

# State/cooldown
COOLDOWN_SEC = int(os.getenv("TB_TRADER_COOLDOWN_SEC", "3600"))
STATE_DIR = Path("state")
RUNS_DIR = Path("runs")

# ==========================
# Helpers
# ==========================

def _rest() -> REST:
    return REST(
        key_id=settings.alpaca_key_id,
        secret_key=settings.alpaca_secret_key,
        base_url=settings.alpaca_base_url,
    )


def _normalize_symbol(sym: str) -> str:
    if "/" in sym:
        return sym
    if len(sym) >= 6:
        return f"{sym[:-3]}/{sym[-3:]}"
    return sym


def _decimals_for(sym: str) -> int:
    # Reasonable default for BTC/USD
    return 2 if "USD" in sym.replace("/", "") else 6


def _state_key_for(sym: str) -> str:
    s = _normalize_symbol(sym).replace("/", "-")
    return f"hybrid_trader_state_{s}.json"


def load_state(symbol: str) -> Dict[str, Any]:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        p = STATE_DIR / _state_key_for(symbol)
        if not p.exists():
            return {}
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(symbol: str, st: Dict[str, Any]) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        p = STATE_DIR / _state_key_for(symbol)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(st, f, indent=2, sort_keys=True)
        tmp.replace(p)
    except Exception as e:
        logger.warning(f"[state] save failed: {e}")


def _nowstamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
        tmp.replace(path)
    except Exception as e:
        logger.warning(f"[audit] write failed: {e}")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def maybe_auto_apply_params(now_utc: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """If enabled and not killed, load promoted params and apply to runtime globals.

    Maps promoted config fields to live runtime variables:
      - risk_frac -> MAX_PORTFOLIO_RISK
      - stop_mode == fixed_pct -> USE_ATR_STOP=False, TP_PCT/SL_PCT from tp_pct/sl_pct
      - stop_mode in {atr_fixed, atr_trailing} -> USE_ATR_STOP=True, ATR_STOP_MULT from atr_mult, ATR_LEN from atr_period, TP_PCT from tp_pct

    Writes an audit JSON under eval_runs/live_auto_apply/.
    """
    # Declare globals up-front since we both read and assign them
    global MAX_PORTFOLIO_RISK, TP_PCT, SL_PCT, USE_ATR_STOP, ATR_STOP_MULT, ATR_LEN
    if not AUTO_APPLY_ENABLED or AUTO_APPLY_KILL:
        return None
    try:
        if not PROMOTED_PARAMS_PATH.exists():
            return None
        with PROMOTED_PARAMS_PATH.open("r") as f:
            cfg = json.load(f)
        applied: Dict[str, Any] = {"status": "skipped", "reason": "no_changes"}
        # Prepare current snapshot
        curr = {
            "MAX_PORTFOLIO_RISK": MAX_PORTFOLIO_RISK,
            "TP_PCT": TP_PCT,
            "SL_PCT": SL_PCT,
            "USE_ATR_STOP": USE_ATR_STOP,
            "ATR_STOP_MULT": ATR_STOP_MULT,
            "ATR_LEN": ATR_LEN,
        }
        # Compute next
        next_vals = dict(curr)
        risk_frac = _safe_float(cfg.get("risk_frac"), curr["MAX_PORTFOLIO_RISK"])
        stop_mode = str(cfg.get("stop_mode") or "").strip()
        tp_pct = cfg.get("tp_pct")
        sl_pct = cfg.get("sl_pct")
        atr_mult = cfg.get("atr_mult")
        atr_period = cfg.get("atr_period")
        if risk_frac is not None:
            next_vals["MAX_PORTFOLIO_RISK"] = float(max(0.0, min(0.05, risk_frac)))
        if stop_mode == "fixed_pct":
            next_vals["USE_ATR_STOP"] = False
            if tp_pct is not None:
                next_vals["TP_PCT"] = max(0.0, float(tp_pct))
            if sl_pct is not None:
                next_vals["SL_PCT"] = max(0.0, float(sl_pct))
        elif stop_mode in ("atr_fixed", "atr_trailing"):
            next_vals["USE_ATR_STOP"] = True
            if atr_mult is not None:
                next_vals["ATR_STOP_MULT"] = max(0.1, float(atr_mult))
            if atr_period is not None:
                try:
                    next_vals["ATR_LEN"] = max(1, int(atr_period))
                except Exception:
                    pass
            if tp_pct is not None:
                next_vals["TP_PCT"] = max(0.0, float(tp_pct))
        # Detect changes
        changed = {k: (curr[k], next_vals[k]) for k in curr.keys() if curr[k] != next_vals[k]}
        if not changed:
            return None
        # Apply to globals
        MAX_PORTFOLIO_RISK = float(next_vals["MAX_PORTFOLIO_RISK"])
        TP_PCT = float(next_vals["TP_PCT"])
        SL_PCT = float(next_vals["SL_PCT"]) if not next_vals["USE_ATR_STOP"] else SL_PCT
        USE_ATR_STOP = bool(next_vals["USE_ATR_STOP"])
        ATR_STOP_MULT = float(next_vals["ATR_STOP_MULT"]) if USE_ATR_STOP else ATR_STOP_MULT
        ATR_LEN = int(next_vals["ATR_LEN"]) if USE_ATR_STOP else ATR_LEN
        # Audit
        ts = now_utc or datetime.now(timezone.utc).isoformat()
        audit = {
            "ts_utc": ts,
            "status": "applied",
            "from": curr,
            "to": next_vals,
            "promoted": cfg,
        }
        AUTO_APPLY_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(AUTO_APPLY_AUDIT_DIR / f"apply_{_nowstamp()}.json", audit)
        logger.info("[auto_apply] Applied promoted params: %s", {k: v[1] for k, v in changed.items()})
        return audit
    except Exception as e:
        try:
            AUTO_APPLY_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
            write_json(AUTO_APPLY_AUDIT_DIR / f"apply_error_{_nowstamp()}.json", {"error": str(e)[:500]})
        except Exception:
            pass
        logger.warning(f"[auto_apply] failed: {e}")
        return None


def fetch_bars(symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
    """Fetch recent crypto bars for given timeframe using Alpaca v2 crypto bars."""
    if OFFLINE:
        return synthetic_bars(timeframe, lookback)
    api = _rest()
    sym = _normalize_symbol(symbol)
    end = datetime.now(timezone.utc)
    # add buffer bars
    if timeframe == "1Min":
        delta = timedelta(minutes=lookback + 5)
    elif timeframe == "15Min":
        delta = timedelta(minutes=(lookback + 4) * 15)
    elif timeframe == "1Hour":
        delta = timedelta(hours=lookback + 2)
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    start = end - delta
    def _get_bars():
        return api.get_crypto_bars(sym, timeframe, start.isoformat(), end.isoformat())
    def _on_retry(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
        logger.warning(f"[retry] get_bars attempt {attempt} status={status} err={str(exc)[:120]} next={sleep_s:.2f}s")
    bars_resp = retry_call(
        _get_bars,
        attempts=int(os.getenv("TB_RETRY_ATTEMPTS", "5")),
        retry_exceptions=(Exception,),
        retry_status_codes=RETRY_STATUS_CODES,
        on_retry=_on_retry,
    )
    bars = bars_resp.df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(sym, level=0)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    else:
        bars = bars.tz_convert("UTC")
    # keep only required columns
    bars = bars[["open", "high", "low", "close", "volume"]].copy()
    bars.sort_index(inplace=True)
    return bars


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def detect_cross_up(fast: pd.Series, slow: pd.Series) -> bool:
    if len(fast) < 2 or len(slow) < 2:
        return False
    f0, f1 = fast.iloc[-2], fast.iloc[-1]
    s0, s1 = slow.iloc[-2], slow.iloc[-1]
    return f0 <= s0 and f1 > s1


def detect_cross_down(fast: pd.Series, slow: pd.Series) -> bool:
    if len(fast) < 2 or len(slow) < 2:
        return False
    f0, f1 = fast.iloc[-2], fast.iloc[-1]
    s0, s1 = slow.iloc[-2], slow.iloc[-1]
    return f0 >= s0 and f1 < s1


def one_hour_trend_up(bars_1h: pd.DataFrame) -> bool:
    ema50 = ema(bars_1h["close"], 50)
    return bool(bars_1h["close"].iloc[-1] > ema50.iloc[-1])

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()


# ==========================
# Perplexity Sentiment
# ==========================

def _pplx_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def sentiment_via_perplexity(headlines: list[str]) -> Tuple[float, Optional[str]]:
    """
    Ask Perplexity to score sentiment in [0,1] for Bitcoin given the provided headlines.
    Rotates through settings.pplx_api_keys. Returns (score, error_or_none).
    """
    if OFFLINE:
        # Deterministic mock sentiment in offline mode; no network calls
        return 0.62, None
    keys = settings.pplx_api_keys or []
    if not keys:
        return 0.0, "No Perplexity API keys"
    # Build a compact prompt; require strict JSON
    system_msg = (
        "You are an analysis API. Respond ONLY with a JSON object: {\"sentiment\": <float 0..1>}"
    )
    joined = "\n".join([h.strip() for h in headlines if h.strip()])[:2000]
    user_msg = (
        "Given these recent Bitcoin headlines, output a single scalar sentiment score in [0,1] "
        "where 0 is strongly bearish and 1 is strongly bullish.\n"
        f"Headlines:\n{joined}"
    )
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "web_search_options": {"enable_citation": False},
    }
    last_err: Optional[str] = None
    with httpx.Client(timeout=PPLX_TIMEOUT) as client:
        for key in keys:
            try:
                def _post() -> httpx.Response:
                    return client.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers=_pplx_headers(key),
                        json=payload,
                    )
                def _on_retry(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
                    logger.warning(
                        f"[retry] pplx attempt {attempt} status={status} err={str(exc)[:120]} next={sleep_s:.2f}s"
                    )
                r = retry_call(
                    _post,
                    attempts=int(os.getenv("TB_RETRY_ATTEMPTS", "5")),
                    retry_exceptions=(httpx.TimeoutException, httpx.TransportError, TimeoutError, ConnectionError),
                    retry_status_codes=RETRY_STATUS_CODES,
                    on_retry=_on_retry,
                )
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code} {r.text[:160]}"; continue
                data = r.json()
                content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
                text = str(content or "").strip()
                # Extract JSON object
                if text.startswith("```"):
                    lines = text.splitlines()
                    if lines and lines[0].startswith("```"): lines = lines[1:]
                    if lines and lines[-1].startswith("```"): lines = lines[:-1]
                    text = "\n".join(lines)
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    obj = {}
                    try:
                        import json
                        obj = json.loads(text[start:end+1])
                    except Exception:
                        pass
                    val = obj.get("sentiment") if isinstance(obj, dict) else None
                    try:
                        score = float(val)
                        if 0.0 <= score <= 1.0:
                            return score, None
                    except Exception:
                        pass
                last_err = "Invalid JSON response"
            except Exception as e:
                last_err = str(e)
    return 0.0, last_err


# ==========================
# Execution
# ==========================

class _MLP(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x)

def _load_ml_gate(model_path: str, features_path: str):
    try:
        import pandas as _pd
        feats = _pd.read_csv(features_path)["feature"].tolist()
        model = _MLP(len(feats))
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, feats
    except Exception as e:
        logger.warning(f"[ml_gate] load failed: {e}")
        return None, None

def _build_live_feature_vector(bars_15: pd.DataFrame, feature_names: list[str]) -> Optional[torch.Tensor]:
    try:
        df = bars_15.copy()

        # Basic returns
        df["ret_1"] = df["close"].pct_change()

        # RSI (Relative Strength Index)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        df["ema12"] = ema(df["close"], 12)
        df["ema26"] = ema(df["close"], 26)
        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = ema(df["macd"], 9)
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        df["sma20"] = df["close"].rolling(window=20).mean()
        df["std20"] = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["sma20"] + (df["std20"] * 2)
        df["bb_lower"] = df["sma20"] - (df["std20"] * 2)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Momentum indicators
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1

        # Volatility measures
        df["close_volatility"] = df["close"].pct_change().rolling(20).std()
        df["high_low_range"] = (df.get("high", df["close"]) - df.get("low", df["close"])) / df["close"]

        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Original features
        df["ema12_slope"] = df["ema12"].pct_change()
        df["ema26_slope"] = df["ema26"].pct_change()
        df["vol"] = df["close"].pct_change().rolling(20).std().fillna(0.0)
        df["vol_chg"] = df["volume"].pct_change().fillna(0.0)
        df["cross_up"] = ((df["ema12"].shift(1) <= df["ema26"].shift(1)) & (df["ema12"] > df["ema26"]))
        df["cross_up"] = df["cross_up"].astype(int)

        # Align to the last fully-formed feature row (like training: features drop last row)
        df = df.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        # Use the penultimate row to mimic training alignment
        last_idx = -2 if len(df) >= 2 else -1
        try:
            vec = [float(df.iloc[last_idx][name]) for name in feature_names]
        except KeyError as e:
            logger.warning(f"[ml_gate] Missing feature: {e}, available: {list(df.columns)}")
            return None
        x = torch.tensor([vec], dtype=torch.float32)
        return x
    except Exception as e:
        logger.warning(f"[ml_gate] feature build failed: {e}")
        return None

def get_account_equity(api: REST) -> float:
    try:
        def _get_acct():
            return api.get_account()
        def _on_retry(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
            logger.warning(f"[retry] get_account attempt {attempt} status={status} err={str(exc)[:120]} next={sleep_s:.2f}s")
        acct = retry_call(
            _get_acct,
            attempts=int(os.getenv("TB_RETRY_ATTEMPTS", "5")),
            retry_exceptions=(Exception,),
            retry_status_codes=RETRY_STATUS_CODES,
            on_retry=_on_retry,
        )
        eq = float(getattr(acct, "equity", getattr(acct, "cash", 0.0)) or 0.0)
        return eq
    except Exception:
        return 0.0


def reconcile_position_state(api: Optional[REST], symbol: str, st: Dict[str, Any]) -> Dict[str, Any]:
    """Update st['in_position'] based on broker positions when online; otherwise keep as-is."""
    if api is None:
        return st
    try:
        sym = _normalize_symbol(symbol)
        qty = 0.0
        for p in api.list_positions():
            if getattr(p, "symbol", "") == sym:
                try:
                    qty = abs(float(p.qty))
                except Exception:
                    qty = 0.0
                break
        st = dict(st)
        st["in_position"] = bool(qty > 0.0)
        return st
    except Exception as e:
        logger.warning(f"[state] reconcile failed: {e}")
        return st


def calc_position_size(equity: float, entry: float, stop: float) -> float:
    risk_amt = equity * MAX_PORTFOLIO_RISK
    per_unit_risk = max(entry - stop, 1e-9)
    qty = risk_amt / per_unit_risk
    # Enforce optional hard notional cap per trade
    try:
        cap_notional = float(os.getenv("TB_MAX_NOTIONAL_PER_TRADE", "0"))
    except Exception:
        cap_notional = 0.0
    if cap_notional > 0:
        max_qty = cap_notional / max(entry, 1e-9)
        qty = min(qty, max_qty)
    return max(0.0, float(qty))


def place_bracket(api: REST, symbol: str, qty: float, entry: float, tp: float, sl: float) -> Tuple[bool, Optional[str], Optional[str]]:
    try:
        # Enforce optional hard notional cap per trade at submission time as a safety net
        try:
            cap_notional = float(os.getenv("TB_MAX_NOTIONAL_PER_TRADE", "0"))
        except Exception:
            cap_notional = 0.0
        if cap_notional > 0 and entry > 0:
            try:
                qty = min(float(qty), cap_notional / float(entry))
            except Exception:
                pass
        def _submit():
            return api.submit_order(
                symbol=_normalize_symbol(symbol),
                side="buy",
                type="market",
                time_in_force="gtc",
                qty=qty,
                take_profit={"limit_price": round(tp, _decimals_for(symbol))},
                stop_loss={"stop_price": round(sl, _decimals_for(symbol))},
            )
        def _on_retry(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
            logger.warning(f"[retry] submit_order attempt {attempt} status={status} err={str(exc)[:120]} next={sleep_s:.2f}s")
        order = retry_call(
            _submit,
            attempts=int(os.getenv("TB_RETRY_ATTEMPTS", "5")),
            retry_exceptions=(Exception,),
            retry_status_codes=RETRY_STATUS_CODES,
            on_retry=_on_retry,
        )
        oid = getattr(order, "id", None) or getattr(order, "client_order_id", None)
        # Per-fill logging: order submission
        try:
            log_order_event(
                "order_submitted",
                symbol=symbol,
                side="buy",
                qty=qty,
                extra={
                    "order_id": (str(oid) if oid else None),
                    "entry": entry,
                    "tp": tp,
                    "sl": sl,
                },
            )
        except Exception:
            pass
        # Best-effort immediate fill check (non-blocking)
        try:
            try_log_fill_once(api, str(oid) if oid else None, symbol, "buy")
        except Exception:
            pass
        return True, (str(oid) if oid else None), None
    except Exception as e:
        return False, None, str(e)


def close_position_if_any(api: REST, symbol: str) -> Optional[str]:
    sym = _normalize_symbol(symbol)
    try:
        def _list_pos():
            return api.list_positions()
        def _on_retry_list(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
            logger.warning(f"[retry] list_positions attempt {attempt} status={status} err={str(exc)[:120]} next={sleep_s:.2f}s")
        positions = retry_call(
            _list_pos,
            attempts=int(os.getenv("TB_RETRY_ATTEMPTS", "5")),
            retry_exceptions=(Exception,),
            retry_status_codes=RETRY_STATUS_CODES,
            on_retry=_on_retry_list,
        )
        pos = None
        for p in positions:
            if getattr(p, "symbol", "") == sym:
                pos = p; break
        if not pos:
            return None
        qty = abs(float(pos.qty))
        if qty <= 0:
            return None
        def _submit_close():
            return api.submit_order(symbol=sym, side="sell", type="market", time_in_force="gtc", qty=qty)
        def _on_retry_close(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
            logger.warning(f"[retry] submit_close attempt {attempt} status={status} err={str(exc)[:120]} next={sleep_s:.2f}s")
        order = retry_call(
            _submit_close,
            attempts=int(os.getenv("TB_RETRY_ATTEMPTS", "5")),
            retry_exceptions=(Exception,),
            retry_status_codes=RETRY_STATUS_CODES,
            on_retry=_on_retry_close,
        )
        oid = getattr(order, "id", None) or getattr(order, "client_order_id", None)
        # Per-fill logging: close order submission
        try:
            log_order_event(
                "order_submitted",
                symbol=sym,
                side="sell",
                qty=qty,
                extra={
                    "order_id": (str(oid) if oid else None),
                    "reason": "close_position",
                },
            )
        except Exception:
            pass
        # Best-effort immediate fill check (non-blocking)
        try:
            try_log_fill_once(api, str(oid) if oid else None, sym, "sell")
        except Exception:
            pass
        return str(oid) if oid else None
    except Exception as e:
        logger.warning(f"[trade] close error: {e}")
        return None


# ==========================
# Notifications
# ==========================

# Import Telegram sender from project root module
try:
    from telegram_bot import send_message as send_telegram  # type: ignore
except Exception as _e:
    # Keep module import-safe; notify() will guard and log if used
    send_telegram = None  # type: ignore

def send_discord_embed(webhook_url: str, embeds: list[dict]) -> bool:
    """Minimal Discord webhook sender. Returns True on HTTP 2xx, else False."""
    try:
        if not webhook_url:
            return False
        with httpx.Client(timeout=float(os.getenv("TB_DISCORD_TIMEOUT", "5"))) as client:
            resp = client.post(webhook_url, json={"embeds": embeds})
            if 200 <= resp.status_code < 300:
                return True
            logger.warning(f"[discord] status={resp.status_code} body={resp.text[:200]}")
            return False
    except Exception as e:
        logger.warning(f"[discord] error: {e}")
        return False

def notify(event: str, payload: Dict[str, Any]) -> None:
    if not NOTIFY:
        return
    # Tailor fields for readability depending on event type; keep TG/Discord parity
    et = event.lower()
    sym = payload.get("symbol", "")
    sent = payload.get("sentiment")
    status = payload.get("status")
    # Build message strings per event
    if et in ("submit", "would_submit"):
        qty = payload.get("qty")
        entry = payload.get("entry")
        tp = payload.get("tp")
        sl = payload.get("sl")
        price = payload.get("price")
        desc_lines = [
            f"tf_fast={TF_FAST} tf_slow={TF_SLOW}",
            f"qty={qty} entry={entry} tp={tp} sl={sl}",
            f"price={price} sentiment={sent}",
            f"status={status}",
        ]
        tg_msg = (
            f"Hybrid Trader • {sym}. {event}. "
            f"qty={qty} entry={entry} tp={tp} sl={sl} sentiment={sent}"
        )
        color = 0x2ecc71
    elif et in ("close", "would_close"):
        qty = payload.get("qty")
        entry = payload.get("entry")
        exit_px = payload.get("price")
        pnl_est = payload.get("pnl_est")
        desc_lines = [
            f"tf_fast={TF_FAST} tf_slow={TF_SLOW}",
            f"qty={qty} entry={entry} exit={exit_px} pnl_est={pnl_est}",
            f"sentiment={sent}",
            f"status={status}",
        ]
        tg_msg = (
            f"Hybrid Trader • {sym}. {event}. "
            f"qty={qty} entry={entry} exit={exit_px} pnl={pnl_est} sentiment={sent}"
        )
        color = 0x95a5a6
    else:  # heartbeat/others
        price = payload.get("price")
        desc_lines = [
            f"tf_fast={TF_FAST} tf_slow={TF_SLOW}",
            f"price={price} sentiment={sent}",
            f"status={status}",
        ]
        tg_msg = (
            f"Hybrid Trader • {sym}. {event}. "
            f"price={price} sentiment={sent} status={status}"
        )
        color = 0x95a5a6

    embed = {
        "title": f"Trader: {event} {sym}",
        "description": "\n".join(desc_lines),
        "color": color,
    }
    if ENABLE_DISCORD:
        try:
            webhook = os.getenv("DISCORD_TRADER_WEBHOOK_URL", "") or os.getenv("DISCORD_WEBHOOK_URL", "")
            if webhook:
                send_discord_embed(webhook, [embed])
        except Exception as e:
            logger.warning(f"[notify] discord error: {e}")
    if not NO_TELEGRAM:
        try:
            msg = tg_msg
            if send_telegram is not None:
                send_telegram(msg)
            else:
                logger.info("[notify] telegram module not available; skipped")
        except Exception as e:
            logger.warning(f"[notify] telegram error: {e}")


# ==========================
# Main loop (single-run)
# ==========================

def main() -> int:
    logger.info("Starting Hybrid EMA+Sentiment Trader (safe=%s, no_trade=%s)", OFFLINE, NO_TRADE)
    # Assign a run_id early for consistent logging regardless of TB_AUDIT
    _run_id = _nowstamp()
    logger.info("[run] start run_id=%s symbol=%s", _run_id, SYMBOL)
    # Log resolved ML model directory for reproducibility and grep-ability
    try:
        if USE_ML_GATE and ML_MODEL_PATH:
            _resolved_model_dir = os.path.dirname(os.path.realpath(ML_MODEL_PATH))
            logger.info("[ml_gate] using model_dir=%s", _resolved_model_dir)
    except Exception:
        pass
    # Attempt to auto-apply promoted parameters (opt-in)
    maybe_auto_apply_params()
    api = _rest() if not OFFLINE else None
    # Load and reconcile state
    state = load_state(SYMBOL)
    if not OFFLINE:
        state = reconcile_position_state(api, SYMBOL, state)
    # Daily PnL book-keeping
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("pnl_date") != today:
        # set equity reference at start of day
        eq_ref = get_account_equity(api) if not OFFLINE else 100000.0
        state.update({"pnl_date": today, "pnl_today": 0.0, "equity_ref": float(eq_ref)})

    logger.info("[progress] Fetching bars...")
    # Fetch bars
    bars_15 = fetch_bars(SYMBOL, TF_FAST, lookback=200)
    bars_1h = fetch_bars(SYMBOL, TF_SLOW, lookback=200)
    if len(bars_15) < 50 or len(bars_1h) < 60:
        logger.warning("Insufficient bars fetched: 15m=%d 1h=%d", len(bars_15), len(bars_1h))
        return 1

    ema12 = ema(bars_15["close"], 12)
    ema26 = ema(bars_15["close"], 26)
    ema50h = ema(bars_1h["close"], 50)
    # 1h signal EMAs
    ema1h_fast = ema(bars_1h["close"], EMA_1H_FAST)
    ema1h_slow = ema(bars_1h["close"], EMA_1H_SLOW)
    cross_up = detect_cross_up(ema12, ema26)
    cross_down = detect_cross_down(ema12, ema26)
    cross_up_1h = detect_cross_up(ema1h_fast, ema1h_slow)
    cross_down_1h = detect_cross_down(ema1h_fast, ema1h_slow)
    trend_up = bool(bars_1h["close"].iloc[-1] > ema50h.iloc[-1])

    logger.info("[progress] Computing indicators...")
    price = float(bars_15["close"].iloc[-1])

    # Sentiment
    # In OFFLINE mode, do not fetch from Alpaca; use mock headlines
    if OFFLINE:
        headlines: List[str] = [
            "Bitcoin consolidates after sharp move; traders eye EMA cross",
            "ETF flows steady as BTC holds key support",
            "Macro stable ahead of Fed speakers; risk tone neutral to positive",
        ]
    else:
        try:
            api_news = _rest()
            def _get_news():
                return api_news.get_news(_normalize_symbol(SYMBOL), limit=10)
            def _on_retry_news(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
                logger.warning(f"[retry] get_news attempt {attempt} status={status} err={str(exc)[:120]} next={sleep_s:.2f}s")
            news = retry_call(
                _get_news,
                attempts=int(os.getenv("TB_RETRY_ATTEMPTS", "5")),
                retry_exceptions=(Exception,),
                retry_status_codes=RETRY_STATUS_CODES,
                on_retry=_on_retry_news,
            )
            headlines = [getattr(n, "headline", getattr(n, "title", "")) for n in news]
            headlines = [h for h in headlines if h]
        except Exception:
            headlines = []
    logger.info("[progress] Fetching sentiment...")
    sentiment, serr = sentiment_via_perplexity(headlines)
    if serr:
        logger.info(f"[sentiment] fallback notice: {serr}")

    # Optional signal debounce: require EMA12>EMA26 across last N bars
    debounce_ok = True
    if SIGNAL_DEBOUNCE_N > 0:
        try:
            cond = (ema12 > ema26).iloc[-SIGNAL_DEBOUNCE_N:]
            debounce_ok = bool(cond.all())
        except Exception:
            debounce_ok = False
    # Optional 1h debounce: require EMA_fast>EMA_slow across last N 1h bars
    debounce_1h_ok = True
    if DEBOUNCE_1H_N > 0:
        try:
            cond1h = (ema1h_fast > ema1h_slow).iloc[-DEBOUNCE_1H_N:]
            debounce_1h_ok = bool(cond1h.all())
        except Exception:
            debounce_1h_ok = False
    logger.info(
        "Signals: 15m[cross_up=%s cross_down=%s debounce_ok=%s] 1h[cross_up=%s cross_down=%s debounce_ok=%s] trend_up=%s sentiment=%.3f price=%.2f",
        cross_up, cross_down, debounce_ok, cross_up_1h, cross_down_1h, debounce_1h_ok, trend_up, sentiment, price,
    )

    # Optional ML probability gate
    ml_prob: Optional[float] = None
    ml_model_dir: Optional[str] = None
    if USE_ML_GATE:
        try:
            ml_model_dir = os.path.dirname(os.path.realpath(ML_MODEL_PATH)) if ML_MODEL_PATH else None
        except Exception:
            ml_model_dir = None
        model, feat_names = _load_ml_gate(ML_MODEL_PATH, ML_FEATURES_PATH)
        if model is not None and feat_names is not None:
            x = _build_live_feature_vector(bars_15, feat_names)
            if x is not None:
                with torch.no_grad():
                    logit = model(x)
                    prob_val = float(torch.sigmoid(logit).item())
                    # Ensure finite numeric
                    if not np.isfinite(prob_val):
                        prob_val = 0.0
                    ml_prob = max(0.0, min(1.0, prob_val))
        # If gate is enabled but we failed to compute, be conservative
        if ml_prob is None:
            ml_prob = 0.0
        logger.info("[ml_gate] prob=%.3f min=%.2f", ml_prob, ML_MIN_PROB)

    # Per-run audit snapshot (inputs + signals + pre-state)
    run_id = _run_id
    run_dir = None
    if os.getenv("TB_AUDIT", "1") == "1":
        run_dir = RUNS_DIR / run_id
        inputs = {
            "symbol": SYMBOL,
            "time": datetime.now(timezone.utc).isoformat(),
            "price": round(price, 2),
            "ema12": float(ema12.iloc[-1]),
            "ema26": float(ema26.iloc[-1]),
            "ema50h": float(ema50h.iloc[-1]),
            "sentiment": float(sentiment),
            "cross_up": bool(cross_up),
            "cross_down": bool(cross_down),
            "cross_up_1h": bool(cross_up_1h),
            "cross_down_1h": bool(cross_down_1h),
            "trend_up": bool(trend_up),
        }
        # Enrich audit with optional gates
        if USE_ML_GATE:
            inputs["ml_prob"] = float(ml_prob if ml_prob is not None else 0.0)
            if ml_model_dir:
                inputs["ml_model_dir"] = ml_model_dir
        # ATR filter info
        atr_pct_val = None
        htf_ok_val = None
        if USE_ATR_FILTER:
            try:
                atr_series = atr(bars_15, ATR_LEN)
                atr_val = float(atr_series.iloc[-1])
                atr_pct_val = float(atr_val / max(price, 1e-9))
            except Exception:
                atr_pct_val = None
        if USE_HTF_REGIME:
            try:
                ema_htf = ema(bars_1h["close"], HTF_EMA_LEN)
                htf_ok_val = bool(bars_1h["close"].iloc[-1] > ema_htf.iloc[-1])
            except Exception:
                htf_ok_val = None
        if atr_pct_val is not None:
            inputs["atr_pct"] = round(atr_pct_val, 6)
        if htf_ok_val is not None:
            inputs["htf_regime_ok"] = bool(htf_ok_val)
        write_json(run_dir / "inputs.json", inputs)
        logger.info(f"[progress] Wrote audit inputs -> {run_dir}/inputs.json")

    # Decision logic
    did_anything = False
    now_ts = int(time.time())
    cooldown_until = int(state.get("cooldown_until", 0))
    in_position = bool(state.get("in_position", False))

    # Optional test hook: force a tiny BUY to validate E2E order flow when enabled.
    # Enabled only when online (OFFLINE=0) and trading allowed (TB_NO_TRADE=0).
    if (os.getenv("TB_TRADER_TEST_FORCE_BUY", "0") == "1") and (not OFFLINE) and (not NO_TRADE):
        try:
            equity = get_account_equity(api) if api is not None else 0.0
        except Exception:
            equity = 0.0
        # ~$10 notional qty (respecting Alpaca ~$10 min)
        test_notional = max(10.0, float(os.getenv("TB_TEST_NOTIONAL", "10")))
        qty = max(0.000001, round(test_notional / max(price, 1e-6), 6))
        entry = float(price)
        tp = entry * (1.0 + TP_PCT)
        # Respect ATR-based stop sizing if enabled
        if USE_ATR_STOP:
            try:
                atr_curr = float(atr(fetch_bars(SYMBOL, TF_FAST, lookback=ATR_LEN + 60), ATR_LEN).iloc[-1]) if OFFLINE else float(atr(bars_15, ATR_LEN).iloc[-1])
            except Exception:
                atr_curr = 0.0
            sl = max(0.0, entry - ATR_STOP_MULT * atr_curr)
        else:
            sl = entry * (1.0 - SL_PCT)
        if api is not None:
            ok, oid, err = place_bracket(api, SYMBOL, qty, entry, tp, sl)
            notify("submit", {"symbol": SYMBOL, "side": "buy", "qty": qty, "entry": round(entry, 2), "tp": round(tp, 2), "sl": round(sl, 2), "sentiment": round(sentiment, 3), "status": "submitted" if ok else f"failed:{err}"})
            logger.info("[test] forced BUY submitted: ok=%s id=%s err=%s", ok, oid, err)
            did_anything = True
        decision = {"action": "close"}

    # Compute optional filter booleans
    atr_ok = True
    if USE_ATR_FILTER:
        try:
            atr_curr = float(atr(bars_15, ATR_LEN).iloc[-1])
            atr_pct = float(atr_curr / max(price, 1e-9))
            atr_ok = (atr_pct >= ATR_MIN_PCT) and (atr_pct <= ATR_MAX_PCT)
        except Exception:
            atr_ok = False
    htf_ok = True
    if USE_HTF_REGIME:
        try:
            ema_htf = ema(bars_1h["close"], HTF_EMA_LEN)
            htf_ok = bool(bars_1h["close"].iloc[-1] > ema_htf.iloc[-1])
        except Exception:
            htf_ok = False

    decision = {"action": "hold", "reason": "no_signal"}
    # Primary 15m entry path
    # Evaluate ML gate condition with optional soft-neutral behavior
    def _ml_gate_ok() -> bool:
        if not USE_ML_GATE:
            return True
        if ml_prob is None:
            return ML_SOFT_GATE  # neutral pass when soft gate enabled
        return ml_prob >= ML_MIN_PROB

    if cross_up and trend_up and debounce_ok and atr_ok and htf_ok and sentiment >= SENTIMENT_THRESHOLD and _ml_gate_ok():
        # Long entry
        # Risk gate: daily loss cap
        loss_cap = -DAILY_LOSS_CAP_PCT * float(state.get("equity_ref", 100000.0))
        if float(state.get("pnl_today", 0.0)) <= loss_cap:
            logger.info("[gate] Entry blocked: daily loss cap reached (pnl_today=%.2f cap=%.2f)", state.get("pnl_today", 0.0), loss_cap)
        elif cooldown_until > now_ts:
            logger.info("[gate] Entry blocked by cooldown (%ds remaining)", cooldown_until - now_ts)
        elif in_position:
            logger.info("[gate] Entry blocked: already in_position")
        elif NO_TRADE or OFFLINE:
            logger.info("[gate] would BUY but blocked by no_trade/offline gates")
        # Compute entry/TP/SL
        entry = float(price)
        tp = entry * (1.0 + TP_PCT)
        if USE_ATR_STOP:
            try:
                atr_curr = float(atr(bars_15, ATR_LEN).iloc[-1])
            except Exception:
                atr_curr = 0.0
            sl = max(0.0, entry - ATR_STOP_MULT * atr_curr)
        else:
            sl = entry * (1.0 - SL_PCT)
        equity = get_account_equity(api) if not OFFLINE else 100000.0
        qty = calc_position_size(equity, entry, sl)
        payload = {"symbol": SYMBOL, "entry": round(entry, 2), "tp": round(tp, 2), "sl": round(sl, 2), "qty": round(qty, 6), "price": round(price, 2), "sentiment": round(sentiment, 3), "status": "preview"}
        if (cooldown_until <= now_ts) and (not in_position):
            if OFFLINE or NO_TRADE:
                logger.info("[preview] Would BUY qty=%.6f @ %.2f tp=%.2f sl=%.2f", qty, entry, tp, sl)
                notify("would_submit", payload)
                # Update state with intent + cooldown start (preview mode still sets cooldown)
                state.update({
                    "in_position": True,
                    "last_entry": float(entry),
                    "last_entry_ts": now_ts,
                    "cooldown_until": now_ts + COOLDOWN_SEC,
                    "last_qty": float(qty),
                })
                did_anything = True
            else:
                ok, oid, err = place_bracket(api, SYMBOL, qty, entry, tp, sl)
                payload.update({"status": "submitted" if ok else f"error: {err}", "order_id": oid})
                notify("submit", payload)
                logger.info("[submit] %s", payload["status"]) 
                if ok:
                    state.update({
                        "in_position": True,
                        "last_entry": float(entry),
                        "last_entry_ts": now_ts,
                        "cooldown_until": now_ts + COOLDOWN_SEC,
                        "last_order_id": oid,
                        "last_qty": float(qty),
                    })
                did_anything = True
            # Mark decision for audit
            decision = {"action": "buy", "qty": float(qty), "entry": float(entry), "tp": float(tp), "sl": float(sl)}

    # Secondary 1h entry path (smaller size) — only if 15m path didn't act
    if (decision.get("action") == "hold") and USE_1H_ENTRY and cross_up_1h and debounce_1h_ok and atr_ok and htf_ok and sentiment >= SENTIMENT_THRESHOLD and _ml_gate_ok():
        # Risk gate: daily loss cap
        loss_cap = -DAILY_LOSS_CAP_PCT * float(state.get("equity_ref", 100000.0))
        if float(state.get("pnl_today", 0.0)) <= loss_cap:
            logger.info("[gate-1h] Entry blocked: daily loss cap reached (pnl_today=%.2f cap=%.2f)", state.get("pnl_today", 0.0), loss_cap)
        elif cooldown_until > now_ts:
            logger.info("[gate-1h] Entry blocked by cooldown (%ds remaining)", cooldown_until - now_ts)
        elif in_position:
            logger.info("[gate-1h] Entry blocked: already in_position")
        elif NO_TRADE or OFFLINE:
            logger.info("[gate-1h] would BUY (1h) but blocked by no_trade/offline gates")
        # Compute entry/TP/SL
        entry = float(price)
        tp = entry * (1.0 + TP_PCT)
        if USE_ATR_STOP:
            try:
                atr_curr = float(atr(bars_15, ATR_LEN).iloc[-1])
            except Exception:
                atr_curr = 0.0
            sl = max(0.0, entry - ATR_STOP_MULT * atr_curr)
        else:
            sl = entry * (1.0 - SL_PCT)
        equity = get_account_equity(api) if not OFFLINE else 100000.0
        qty_base = calc_position_size(equity, entry, sl)
        qty = max(0.0, float(qty_base * max(0.0, min(1.0, SIZE_1H_MULT))))
        payload = {"symbol": SYMBOL, "entry": round(entry, 2), "tp": round(tp, 2), "sl": round(sl, 2), "qty": round(qty, 6), "price": round(price, 2), "sentiment": round(sentiment, 3), "status": "preview", "path": "1h"}
        if (cooldown_until <= now_ts) and (not in_position):
            if OFFLINE or NO_TRADE:
                logger.info("[preview-1h] Would BUY(1h) qty=%.6f @ %.2f tp=%.2f sl=%.2f", qty, entry, tp, sl)
                notify("would_submit", payload)
                # Update state with intent + cooldown start (preview mode still sets cooldown)
                state.update({
                    "in_position": True,
                    "last_entry": float(entry),
                    "last_entry_ts": now_ts,
                    "cooldown_until": now_ts + COOLDOWN_SEC,
                    "last_qty": float(qty),
                })
                did_anything = True
            else:
                ok, oid, err = place_bracket(api, SYMBOL, qty, entry, tp, sl)
                payload.update({"status": "submitted" if ok else f"error: {err}", "order_id": oid})
                notify("submit", payload)
                logger.info("[submit-1h] %s", payload["status"]) 
                if ok:
                    state.update({
                        "in_position": True,
                        "last_entry": float(entry),
                        "last_entry_ts": now_ts,
                        "cooldown_until": now_ts + COOLDOWN_SEC,
                        "last_order_id": oid,
                        "last_qty": float(qty),
                    })
                did_anything = True
            decision = {"action": "buy_1h", "qty": float(qty), "entry": float(entry), "tp": float(tp), "sl": float(sl)}

    # Exit condition: bearish cross — notify only if currently in a position
    if cross_down:
        if not state.get("in_position"):
            logger.info("[close] skipped: not in_position")
        elif OFFLINE or NO_TRADE:
            logger.info("[preview] Would CLOSE open position due to bearish cross")
            # Estimate PnL for preview close if we have last_entry/qty
            last_entry = state.get("last_entry")
            last_qty = state.get("last_qty")
            pnl = 0.0
            try:
                if (last_entry is not None) and (last_qty is not None):
                    pnl = (float(price) - float(last_entry)) * float(last_qty)
            except Exception:
                pnl = 0.0
            notify("would_close", {
                "symbol": SYMBOL,
                "price": round(price, 2),  # exit
                "entry": round(float(last_entry), 2) if last_entry is not None else None,
                "qty": float(last_qty) if last_qty is not None else None,
                "pnl_est": round(float(pnl), 2),
                "sentiment": round(sentiment, 3),
                "status": "preview",
            })
            # Preview: clear position and start cooldown
            state.update({
                "in_position": False,
                "last_exit_ts": now_ts,
                "cooldown_until": now_ts + COOLDOWN_SEC,
                "pnl_today": float(state.get("pnl_today", 0.0)) + float(pnl),
            })
            did_anything = True
        else:
            oid = close_position_if_any(api, SYMBOL)
            # Estimate realized PnL using current price vs last_entry for last_qty
            last_entry = state.get("last_entry")
            last_qty = state.get("last_qty")
            pnl = 0.0
            try:
                if (last_entry is not None) and (last_qty is not None):
                    pnl = (float(price) - float(last_entry)) * float(last_qty)
            except Exception:
                pnl = 0.0
            notify("close", {
                "symbol": SYMBOL,
                "price": round(price, 2),  # exit
                "entry": round(float(last_entry), 2) if last_entry is not None else None,
                "qty": float(last_qty) if last_qty is not None else None,
                "pnl_est": round(float(pnl), 2),
                "sentiment": round(sentiment, 3),
                "status": "submitted",
                "order_id": oid,
            })
            logger.info("[close] submitted market close order: %s", oid)
            state.update({
                "in_position": False,
                "last_exit_ts": now_ts,
                "cooldown_until": now_ts + COOLDOWN_SEC,
                "last_close_order_id": oid,
                "pnl_today": float(state.get("pnl_today", 0.0)) + float(pnl),
            })
            did_anything = True

    if not did_anything:
        logger.info("No action taken.")
        decision = {"action": "hold"}
    # Heartbeat: per-run counter + optional liveness notification
    try:
        hb_runs = int(state.get("hb_runs", 0)) + 1
        state["hb_runs"] = hb_runs
        state["last_run_ts"] = now_ts
        if HEARTBEAT and NOTIFY and HEARTBEAT_EVERY_N > 0 and (hb_runs % HEARTBEAT_EVERY_N == 0):
            payload = {
                "symbol": SYMBOL,
                "price": round(price, 2),
                "sentiment": round(sentiment, 3),
                "qty": 0,
                "entry": None,
                "tp": None,
                "sl": None,
                "status": f"alive run={hb_runs}",
            }
            notify("heartbeat", payload)
            state["last_heartbeat_ts"] = now_ts
            logger.info("[heartbeat] sent run=%d every=%d", hb_runs, HEARTBEAT_EVERY_N)
    except Exception as e:
        logger.warning(f"[heartbeat] failed: {e}")

    # Persist state if we touched it
    try:
        save_state(SYMBOL, state)
    except Exception:
        pass

    # Write decision + post-state
    if os.getenv("TB_AUDIT", "1") == "1" and run_dir is not None:
        try:
            write_json(run_dir / "decision.json", {
                "decision": decision,
                "state": state,
            })
            logger.info(f"[progress] Wrote audit decision -> {run_dir}/decision.json")
        except Exception as e:
            logger.warning(f"[audit] decision write failed: {e}")

    # Optional: auto-commit safe artifacts (non-code) using autocommit helper
    try:
        if os.getenv("TB_AUTOCOMMIT_ARTIFACTS", "1") == "1":
            push_enabled = os.getenv("TB_AUTOCOMMIT_PUSH", "1") == "1"
            # Call autocommit.auto_commit_and_push safely
            code = subprocess.call([
                "python3", "-c",
                (
                    "import autocommit as ac; "
                    "print(ac.auto_commit_and_push(['runs','eval_runs','universe_runs','trader_loop.log','trading_agent.log'], "
                    "extra_message='local artifacts', push_enabled="
                    + ("True" if push_enabled else "False") +
                    "))"
                )
            ])
            logger.info("[autocommit] attempted with push=%s status=%s", push_enabled, code)
    except Exception as e:
        logger.warning("[autocommit] failed: %s", e)
    # Final run-complete marker
    try:
        logger.info("[run] complete run_id=%s decision=%s", run_id, (decision or {}).get("action"))
    except Exception:
        pass
    return 0


# ==========================
# Offline helpers
# ==========================
def synthetic_bars(timeframe: str, lookback: int) -> pd.DataFrame:
    """
    Generate deterministic synthetic OHLCV series for preview without network calls.
    Creates a gentle uptrend with minor noise to allow cross/EMA computations.
    """
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    if timeframe == "1Min":
        step = timedelta(minutes=1)
    elif timeframe == "15Min":
        step = timedelta(minutes=15)
    elif timeframe == "1Hour":
        step = timedelta(hours=1)
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    n = max(lookback + 60, 120)  # ensure enough warmup for EMAs
    idx = [now - step * (n - i) for i in range(n)]
    base = 60000.0  # starting price
    drift = 0.0002  # per bar drift
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 5.0, size=n)
    close = base + np.cumsum(base * drift + noise)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.0, 10.0, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.0, 10.0, size=n)
    vol = rng.uniform(5, 50, size=n)
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    }, index=pd.DatetimeIndex(idx, tz="UTC"))
    df = df.iloc[-(lookback + 5):].copy()
    return df

if __name__ == "__main__":
    raise SystemExit(main())
