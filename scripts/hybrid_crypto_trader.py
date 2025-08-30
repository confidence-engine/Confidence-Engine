import os
import sys
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import httpx

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

SYMBOL = os.getenv("SYMBOL", settings.symbol or "BTC/USD")
TF_FAST = "15Min"
TF_SLOW = "1Hour"

# Risk and brackets
MAX_PORTFOLIO_RISK = float(os.getenv("TB_MAX_RISK_FRAC", "0.01"))   # 1%
TP_PCT = float(os.getenv("TB_TP_PCT", "0.05"))                      # +5%
SL_PCT = float(os.getenv("TB_SL_PCT", "0.02"))                      # -2%

# Sentiment
SENTIMENT_THRESHOLD = float(os.getenv("TB_SENTIMENT_CUTOFF", "0.5"))
PPLX_TIMEOUT = float(os.getenv("TB_PPLX_TIMEOUT", "12"))

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
    bars = api.get_crypto_bars(sym, timeframe, start.isoformat(), end.isoformat()).df
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
                r = client.post("https://api.perplexity.ai/chat/completions", headers=_pplx_headers(key), json=payload)
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

def get_account_equity(api: REST) -> float:
    try:
        acct = api.get_account()
        eq = float(getattr(acct, "equity", getattr(acct, "cash", 0.0)) or 0.0)
        return eq
    except Exception:
        return 0.0


def calc_position_size(equity: float, entry: float, stop: float) -> float:
    risk_amt = equity * MAX_PORTFOLIO_RISK
    per_unit_risk = max(entry - stop, 1e-9)
    qty = risk_amt / per_unit_risk
    return max(0.0, float(qty))


def place_bracket(api: REST, symbol: str, qty: float, entry: float, tp: float, sl: float) -> Tuple[bool, Optional[str], Optional[str]]:
    try:
        order = api.submit_order(
            symbol=_normalize_symbol(symbol),
            side="buy",
            type="market",
            time_in_force="gtc",
            qty=qty,
            take_profit={"limit_price": round(tp, _decimals_for(symbol))},
            stop_loss={"stop_price": round(sl, _decimals_for(symbol))},
        )
        oid = getattr(order, "id", None) or getattr(order, "client_order_id", None)
        return True, (str(oid) if oid else None), None
    except Exception as e:
        return False, None, str(e)


def close_position_if_any(api: REST, symbol: str) -> Optional[str]:
    sym = _normalize_symbol(symbol)
    try:
        pos = None
        for p in api.list_positions():
            if getattr(p, "symbol", "") == sym:
                pos = p; break
        if not pos:
            return None
        qty = abs(float(pos.qty))
        if qty <= 0:
            return None
        order = api.submit_order(symbol=sym, side="sell", type="market", time_in_force="gtc", qty=qty)
        oid = getattr(order, "id", None) or getattr(order, "client_order_id", None)
        return str(oid) if oid else None
    except Exception as e:
        logger.warning(f"[trade] close error: {e}")
        return None


# ==========================
# Notifications
# ==========================

def notify(event: str, payload: Dict[str, Any]) -> None:
    if not NOTIFY:
        return
    # Discord embed parity with existing sender
    embed = {
        "title": f"Trader: {event} {payload.get('symbol','')}",
        "description": "\n".join([
            f"tf_fast={TF_FAST} tf_slow={TF_SLOW}",
            f"entry={payload.get('entry')} tp={payload.get('tp')} sl={payload.get('sl')}",
            f"qty={payload.get('qty')} price={payload.get('price')} sentiment={payload.get('sentiment')}",
            f"status={payload.get('status')}",
        ]),
        "color": 0x2ecc71 if event.lower() in ("submit", "would_submit") else 0x95a5a6,
    }
    if ENABLE_DISCORD:
        try:
            webhook = os.getenv("DISCORD_TRADER_WEBHOOK_URL", "") or os.getenv("DISCORD_WEBHOOK_URL", "")
            if webhook:
                send_discord_digest_to(webhook, [embed])
        except Exception as e:
            logger.warning(f"[notify] discord error: {e}")
    if not NO_TELEGRAM:
        try:
            msg = f"Hybrid Trader • {payload.get('symbol','')}. {event}. qty={payload.get('qty')} entry={payload.get('entry')} tp={payload.get('tp')} sl={payload.get('sl')} sentiment={payload.get('sentiment')}"
            send_telegram(msg)
        except Exception as e:
            logger.warning(f"[notify] telegram error: {e}")


# ==========================
# Main loop (single-run)
# ==========================

def main() -> int:
    logger.info("Starting Hybrid EMA+Sentiment Trader (safe=%s, no_trade=%s)", OFFLINE, NO_TRADE)
    api = _rest() if not OFFLINE else None

    # Fetch bars
    bars_15 = fetch_bars(SYMBOL, TF_FAST, lookback=200)
    bars_1h = fetch_bars(SYMBOL, TF_SLOW, lookback=200)
    if len(bars_15) < 50 or len(bars_1h) < 60:
        logger.warning("Insufficient bars fetched: 15m=%d 1h=%d", len(bars_15), len(bars_1h))
        return 1

    ema12 = ema(bars_15["close"], 12)
    ema26 = ema(bars_15["close"], 26)
    cross_up = detect_cross_up(ema12, ema26)
    cross_down = detect_cross_down(ema12, ema26)
    trend_up = one_hour_trend_up(bars_1h)

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
            headlines = [getattr(n, "headline", getattr(n, "title", "")) for n in _rest().get_news(_normalize_symbol(SYMBOL), limit=10)]
            headlines = [h for h in headlines if h]
        except Exception:
            headlines = []
    sentiment, serr = sentiment_via_perplexity(headlines)
    if serr:
        logger.info(f"[sentiment] fallback notice: {serr}")

    logger.info("Signals: cross_up=%s cross_down=%s trend_up=%s sentiment=%.3f price=%.2f", cross_up, cross_down, trend_up, sentiment, price)

    # Decision logic
    did_anything = False

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
        entry = price
        tp = entry * (1.0 + TP_PCT)
        sl = entry * (1.0 - SL_PCT)
        if api is not None:
            ok, oid, err = place_bracket(api, SYMBOL, qty, entry, tp, sl)
            notify("submit", {"symbol": SYMBOL, "side": "buy", "qty": qty, "entry": round(entry, 2), "tp": round(tp, 2), "sl": round(sl, 2), "sentiment": round(sentiment, 3), "status": "submitted" if ok else f"failed:{err}"})
            logger.info("[test] forced BUY submitted: ok=%s id=%s err=%s", ok, oid, err)
            did_anything = True

    if cross_up and trend_up and sentiment >= SENTIMENT_THRESHOLD:
        # Long entry
        if NO_TRADE or OFFLINE:
            logger.info("[gate] would BUY but blocked by no_trade/offline gates")
        equity = get_account_equity(api) if not OFFLINE else 100000.0
        qty = calc_position_size(equity, entry, sl)
        payload = {"symbol": SYMBOL, "entry": round(entry, 2), "tp": round(tp, 2), "sl": round(sl, 2), "qty": round(qty, 6), "price": round(price, 2), "sentiment": round(sentiment, 3), "status": "preview"}
        if OFFLINE or NO_TRADE:
            logger.info("[preview] Would BUY qty=%.6f @ %.2f tp=%.2f sl=%.2f", qty, entry, tp, sl)
            notify("would_submit", payload)
            did_anything = True
        else:
            ok, oid, err = place_bracket(api, SYMBOL, qty, entry, tp, sl)
            payload.update({"status": "submitted" if ok else f"error: {err}", "order_id": oid})
            notify("submit", payload)
            logger.info("[submit] %s", payload["status"]) 
            did_anything = True

    # Exit condition: bearish cross and we have a position
    if cross_down:
        if OFFLINE or NO_TRADE:
            logger.info("[preview] Would CLOSE any open position due to bearish cross")
            notify("would_close", {"symbol": SYMBOL, "price": round(price, 2), "sentiment": round(sentiment, 3), "status": "preview"})
            did_anything = True
        else:
            oid = close_position_if_any(api, SYMBOL)
            notify("close", {"symbol": SYMBOL, "price": round(price, 2), "sentiment": round(sentiment, 3), "status": "submitted", "order_id": oid})
            logger.info("[close] submitted market close order: %s", oid)
            did_anything = True

    if not did_anything:
        logger.info("No action taken.")
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
