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

SYMBOL = os.getenv("SYMBOL", settings.symbol or "BTC/USD")
TF_FAST = "15Min"
TF_SLOW = "1Hour"

# Risk and brackets
MAX_PORTFOLIO_RISK = float(os.getenv("TB_MAX_RISK_FRAC", "0.01"))   # 1%
TP_PCT = float(os.getenv("TB_TP_PCT", "0.05"))                      # +5%
SL_PCT = float(os.getenv("TB_SL_PCT", "0.02"))                      # -2%
DAILY_LOSS_CAP_PCT = float(os.getenv("TB_DAILY_LOSS_CAP_PCT", "0.03"))  # 3% of reference equity

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
    return max(0.0, float(qty))


def place_bracket(api: REST, symbol: str, qty: float, entry: float, tp: float, sl: float) -> Tuple[bool, Optional[str], Optional[str]]:
    try:
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
                send_discord_embed(webhook, [embed])
        except Exception as e:
            logger.warning(f"[notify] discord error: {e}")
    if not NO_TELEGRAM:
        try:
            msg = f"Hybrid Trader • {payload.get('symbol','')}. {event}. qty={payload.get('qty')} entry={payload.get('entry')} tp={payload.get('tp')} sl={payload.get('sl')} sentiment={payload.get('sentiment')}"
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
    cross_up = detect_cross_up(ema12, ema26)
    cross_down = detect_cross_down(ema12, ema26)
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

    logger.info("Signals: cross_up=%s cross_down=%s trend_up=%s sentiment=%.3f price=%.2f", cross_up, cross_down, trend_up, sentiment, price)

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
            "trend_up": bool(trend_up),
        }
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
        entry = price
        tp = entry * (1.0 + TP_PCT)
        sl = entry * (1.0 - SL_PCT)
        if api is not None:
            ok, oid, err = place_bracket(api, SYMBOL, qty, entry, tp, sl)
            notify("submit", {"symbol": SYMBOL, "side": "buy", "qty": qty, "entry": round(entry, 2), "tp": round(tp, 2), "sl": round(sl, 2), "sentiment": round(sentiment, 3), "status": "submitted" if ok else f"failed:{err}"})
            logger.info("[test] forced BUY submitted: ok=%s id=%s err=%s", ok, oid, err)
            did_anything = True
        decision = {"action": "close"}

    decision = {"action": "hold", "reason": "no_signal"}
    if cross_up and trend_up and sentiment >= SENTIMENT_THRESHOLD:
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

    # Exit condition: bearish cross and we have a position
    if cross_down:
        if OFFLINE or NO_TRADE:
            logger.info("[preview] Would CLOSE any open position due to bearish cross")
            notify("would_close", {"symbol": SYMBOL, "price": round(price, 2), "sentiment": round(sentiment, 3), "status": "preview"})
            # Preview: clear position and start cooldown
            # Estimate PnL for preview close if we have last_entry/qty
            pnl = 0.0
            try:
                if state.get("in_position") and (state.get("last_entry") is not None) and (state.get("last_qty") is not None):
                    pnl = (float(price) - float(state.get("last_entry", 0.0))) * float(state.get("last_qty", 0.0))
            except Exception:
                pnl = 0.0
            state.update({
                "in_position": False,
                "last_exit_ts": now_ts,
                "cooldown_until": now_ts + COOLDOWN_SEC,
                "pnl_today": float(state.get("pnl_today", 0.0)) + float(pnl),
            })
            did_anything = True
        else:
            oid = close_position_if_any(api, SYMBOL)
            notify("close", {"symbol": SYMBOL, "price": round(price, 2), "sentiment": round(sentiment, 3), "status": "submitted", "order_id": oid})
            logger.info("[close] submitted market close order: %s", oid)
            # Estimate realized PnL using current price vs last_entry for last_qty
            pnl = 0.0
            try:
                if state.get("in_position") and (state.get("last_entry") is not None) and (state.get("last_qty") is not None):
                    pnl = (float(price) - float(state.get("last_entry", 0.0))) * float(state.get("last_qty", 0.0))
            except Exception:
                pnl = 0.0
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
