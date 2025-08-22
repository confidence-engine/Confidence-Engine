#!/usr/bin/env python3
"""
Crypto Signals → Alpaca (paper) trader
- Consumes the crypto signals digest and prepares bracket orders per asset/TF.
- Safe-by-default: offline and no-trade unless gates are flipped.

Gates (env):
- TB_TRADER_OFFLINE=1  # do not hit Alpaca API; log only
- TB_NO_TRADE=1        # even if online, do not submit orders

Alpaca creds (env):
- ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, ALPACA_BASE_URL

Usage:
- Dry-run (offline, safe):
  TB_TRADER_OFFLINE=1 TB_NO_TRADE=1 \
  python3 scripts/crypto_signals_trader.py --max-coins 6 --tf 1h --debug

- Paper live (will call Alpaca, still gated by TB_NO_TRADE):
  ALPACA_API_KEY_ID=... ALPACA_API_SECRET_KEY=... ALPACA_BASE_URL=https://paper-api.alpaca.markets \
  python3 scripts/crypto_signals_trader.py --max-coins 6 --tf 1h --debug

Notes:
- This initial version uses plan prices embedded in the digest (pre-computed). A 
  live price trigger check can be added later.
"""
from __future__ import annotations
import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta

# Project path
_THIS_DIR = Path(__file__).parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# Local imports
from scripts.crypto_signals_digest import _load_dotenv_if_present, _latest_universe_file, _build_digest_data  # type: ignore
from scripts.discord_sender import send_discord_digest_to  # type: ignore
from alpaca import recent_bars as alpaca_recent_bars  # price helper
import requests

try:
    from alpaca_trade_api.rest import REST
except Exception as _e:  # pragma: no cover
    if os.getenv("TB_TRADER_IMPORT_DEBUG", "0") == "1":
        print(f"[trader] alpaca_trade_api import error: {_e}")
    REST = None  # type: ignore


def _bool(env: str, default: bool = False) -> bool:
    v = os.getenv(env)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _provenance() -> Dict[str, Any]:
    try:
        import subprocess
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(_PROJ_ROOT)).decode().strip()
    except Exception:
        git_sha = None
    return {"git": git_sha}


def _get_alpaca() -> Optional[REST]:
    if REST is None:
        return None
    key = os.getenv("ALPACA_API_KEY_ID", "").strip()
    sec = os.getenv("ALPACA_API_SECRET_KEY", "").strip()
    url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip()
    # Normalize base URL to avoid SDK path duplication like '/v2/v2/account'
    # Accept inputs like '.../v2' or '.../v2/' and strip the version segment.
    try:
        u = url.rstrip("/")
        if u.endswith("/v2"):
            u = u[:-3]  # remove '/v2'
        url = u
    except Exception:
        pass
    if not key or not sec:
        return None
    try:
        return REST(key_id=key, secret_key=sec, base_url=url)
    except Exception:
        return None


def _broker_supports_crypto_shorts() -> bool:
    """Return whether the connected broker supports shorting spot crypto.
    Alpaca spot crypto does NOT support shorting, so this returns False.
    """
    return False


def _calc_qty(equity: float, entry: float, stop: float, risk_frac: float = 0.005) -> float:
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        return 0.0
    return max(0.0, (equity * risk_frac) / per_unit_risk)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _state_path() -> Path:
    p = _PROJ_ROOT / "state"
    p.mkdir(exist_ok=True)
    return p / "crypto_trader_state.json"


def _journal_path() -> Path:
    p = _PROJ_ROOT / "state"
    p.mkdir(exist_ok=True)
    return p / "trade_journal.csv"


def _journal_append(row: Dict[str, Any]) -> None:
    import csv
    jp = _journal_path()
    headers = [
        "ts","event","artifact","symbol","tf","side","bias","entry","stop","tp","price","qty",
        "risk_frac","cooldown_sec","order_id","status","note"
    ]
    exists = jp.exists()
    with jp.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in headers})


def _notify_discord(event: str, payload: Dict[str, Any], debug: bool = False) -> None:
    if os.getenv("TB_TRADER_NOTIFY", "0") != "1":
        return
    webhook = os.getenv("DISCORD_TRADER_WEBHOOK_URL", "").strip()
    if not webhook:
        if debug:
            print("[notify] DISCORD_TRADER_WEBHOOK_URL not set")
        return
    title = f"Trader: {event} {payload.get('side','').upper()} {payload.get('symbol','')}"
    desc_lines = [
        f"tf={payload.get('tf')} qty={payload.get('qty')}",
        f"entry={payload.get('entry')} tp={payload.get('tp')} sl={payload.get('stop')}",
        f"price={payload.get('price')} risk={payload.get('risk_frac')} cooldown={payload.get('cooldown_sec')}s",
        f"artifact={payload.get('artifact')} status={payload.get('status')}"
    ]
    embed = {
        "title": title,
        "description": "\n".join([str(x) for x in desc_lines if x is not None]),
        "color": 0x2ecc71 if event.lower() in ("submit","would_submit") else 0x95a5a6,
    }
    try:
        send_discord_digest_to(webhook, [embed])
    except Exception as e:
        if debug:
            print(f"[notify] error: {e}")


def _load_state() -> Dict[str, Any]:
    sp = _state_path()
    try:
        return json.loads(sp.read_text()) if sp.exists() else {}
    except Exception:
        return {}


def _save_state(st: Dict[str, Any]) -> None:
    sp = _state_path()
    try:
        sp.write_text(json.dumps(st, indent=2))
    except Exception:
        pass


def _cooldown_ok(st: Dict[str, Any], symbol: str, cooldown_sec: int) -> bool:
    rec = (st.get("symbols") or {}).get(symbol)
    if not rec:
        return True
    last = rec.get("last_action_ts")
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
    except Exception:
        return True
    return (_now_utc() - last_dt) >= timedelta(seconds=int(cooldown_sec))


def _mark_action(st: Dict[str, Any], symbol: str) -> None:
    symbols = st.setdefault("symbols", {})
    symbols.setdefault(symbol, {})["last_action_ts"] = _now_utc().isoformat()


def _normalize_symbol(sym: str) -> str:
    s = (sym or "").upper()
    if "/" in s:
        return s
    if len(s) >= 6:
        base, quote = s[:-3], s[-3:]
        return f"{base}/{quote}"
    return s


def _mid_of_zone(entries: Any, side: str) -> Optional[float]:
    # Support mid-of-zone selection for entry triggers when entries represent a zone
    if isinstance(entries, (list, tuple)) and len(entries) == 2 and all(isinstance(x, (int, float)) for x in entries):
        lo, hi = float(entries[0]), float(entries[1])
        return (lo + hi) / 2.0
    if isinstance(entries, (list, tuple)) and len(entries) > 0 and isinstance(entries[0], dict):
        z = entries[0].get("zone_or_trigger")
        if isinstance(z, (list, tuple)) and len(z) == 2 and all(isinstance(x, (int, float)) for x in z):
            lo, hi = float(z[0]), float(z[1])
            return (lo + hi) / 2.0
    return None


def _decimals_for(sym: str) -> int:
    # Simple heuristic for crypto prices; refine per-symbol later
    base = (sym or "").split("/")[0]
    if base in ("BTC", "ETH"):
        return 2
    return 4


def _place_bracket(api: REST, symbol: str, side: str, qty: float, entry: float, tp: float, sl: float, debug: bool = False) -> Tuple[bool, Optional[str], Optional[str]]:
    try:
        # Use literal strings for broad SDK compatibility
        tif = "gtc"
        otype = "market"
        order = api.submit_order(
            symbol=_normalize_symbol(symbol),
            side=side,
            type=otype,
            time_in_force=tif,
            qty=qty,
            take_profit={"limit_price": round(tp, _decimals_for(symbol))},
            stop_loss={"stop_price": round(sl, _decimals_for(symbol))},
        )
        oid = getattr(order, "id", None) or getattr(order, "client_order_id", None)
        if debug:
            print(f"[trade] submitted bracket: sym={symbol} side={side} qty={qty:.6f} tp={tp:.4f} sl={sl:.4f} id={oid}")
        return True, (str(oid) if oid else None), None
    except Exception as e:
        if debug:
            print(f"[trade] submit failed: {e}")
        return False, None, str(e)


def _decide_side(thesis: Dict[str, Any]) -> Optional[str]:
    bias = str(thesis.get("bias", "")).lower()
    if bias == "up":
        return "buy"
    if bias == "down":
        return "sell"
    return None


def _iterate_candidates(digest: Dict[str, Any], tf: str, allowed_symbols: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a in (digest.get("assets") or []):
        sym = a.get("symbol")
        if allowed_symbols is not None:
            # Match using uppercase normalized form
            nsym = _normalize_symbol(sym or "").upper()
            if nsym not in allowed_symbols:
                continue
        plan = a.get("plan") or {}
        thesis = a.get("thesis") or a.get("thesis_text") or {}
        if not sym or tf not in plan:
            continue
        tfp = plan[tf]
        entries = tfp.get("entries")
        invalid = (tfp.get("invalidation") or {}).get("price")
        targets = tfp.get("targets") or []
        if entries is None or invalid is None or not targets:
            continue
        side = _decide_side(a.get("thesis") or {})
        if not side:
            continue
        # Pick TP1
        tp1 = None
        if isinstance(targets, (list, tuple)) and len(targets) > 0:
            t0 = targets[0]
            if isinstance(t0, (int, float)):
                tp1 = float(t0)
            elif isinstance(t0, dict) and isinstance(t0.get("price"), (int, float)):
                tp1 = float(t0["price"])
        if tp1 is None:
            continue
        # entries might be:
        # - float
        # - [lo, hi]
        # - list of dicts with key 'zone_or_trigger' that is float or [lo, hi]
        entry_price: Optional[float] = None
        if isinstance(entries, (int, float)):
            entry_price = float(entries)
        elif isinstance(entries, (list, tuple)):
            if len(entries) == 2 and all(isinstance(x, (int, float)) for x in entries):
                lo, hi = entries
                entry_price = float(lo if side == "sell" else hi)
            elif len(entries) > 0 and isinstance(entries[0], dict):
                e0 = entries[0]
                z = e0.get("zone_or_trigger")
                if isinstance(z, (int, float)):
                    entry_price = float(z)
                elif isinstance(z, (list, tuple)) and len(z) == 2 and all(isinstance(x, (int, float)) for x in z):
                    lo, hi = z
                    entry_price = float(lo if side == "sell" else hi)
        if entry_price is None:
            continue
        out.append({
            "symbol": sym,
            "side": side,
            "entry": float(entry_price),
            "stop": float(invalid),
            "tp": float(tp1),
            "raw_entries": entries,
            "why": tfp.get("explain") or "",
        })
    return out


def _current_price(symbol: str) -> Optional[float]:
    # Try Alpaca with given symbol (may include slash)
    try:
        df = alpaca_recent_bars(symbol, minutes=5)
        if df is not None and len(df) > 0 and "close" in df.columns:
            return float(df["close"].iloc[-1])
    except Exception:
        pass
    # Try Alpaca with normalized symbol (ensure slash form) and also noslash form
    try:
        nsym = _normalize_symbol(symbol)
        if nsym != symbol:
            df = alpaca_recent_bars(nsym, minutes=5)
            if df is not None and len(df) > 0 and "close" in df.columns:
                return float(df["close"].iloc[-1])
        noslash = nsym.replace("/", "")
        df = alpaca_recent_bars(noslash, minutes=5)
        if df is not None and len(df) > 0 and "close" in df.columns:
            return float(df["close"].iloc[-1])
    except Exception:
        pass
    # Binance fallback (maps USD/USDC → USDT)
    try:
        bsym = _map_to_binance(symbol)
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={bsym}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            px = r.json().get("price")
            return float(px) if px is not None else None
    except Exception:
        pass
    return None

def _map_to_binance(sym: str) -> str:
    s = (sym or "").upper()
    base, _, quote = s.partition("/")
    q = quote or "USDT"
    if q in ("USD", "USDC"):
        q = "USDT"
    return f"{base}{q}"


def _has_open_in_direction(api: REST, symbol: str, side: str) -> bool:
    # Check existing position
    try:
        poss = api.list_positions()
        for p in poss or []:
            sym = getattr(p, "symbol", "") or getattr(p, "asset_id", "")
            if sym and sym.upper() == _normalize_symbol(symbol).upper():
                qty = float(getattr(p, "qty", 0.0) or getattr(p, "qty_available", 0.0) or 0.0)
                if qty > 0 and side == "buy":
                    return True
                if qty < 0 and side == "sell":
                    return True
    except Exception:
        pass
    # Check open orders
    try:
        orders = api.list_orders(status="open", nested=True)
        for o in orders or []:
            sym = getattr(o, "symbol", "")
            if sym and sym.upper() == _normalize_symbol(symbol).upper():
                oside = str(getattr(o, "side", "")).lower()
                if oside == side:
                    return True
    except Exception:
        pass
    return False


def _position_qty(api: REST, symbol: str) -> float:
    """Return current base position quantity for symbol (0.0 if none).
    Tries to match both slash and non-slash symbol forms returned by Alpaca.
    """
    try:
        target = _normalize_symbol(symbol).upper()
        target_noslash = target.replace("/", "")
        poss = api.list_positions()
        for p in poss or []:
            sym = (getattr(p, "symbol", "") or getattr(p, "asset_id", "")).upper()
            if not sym:
                continue
            if sym == target or sym == target_noslash:
                q = getattr(p, "qty", None)
                if q is None:
                    q = getattr(p, "qty_available", 0.0)
                try:
                    return float(q)
                except Exception:
                    return 0.0
    except Exception:
        return 0.0
    return 0.0


def _cancel_stale_orders(api: REST, ttl_min: int, debug: bool = False) -> None:
    try:
        from datetime import datetime, timezone
        ttl = timedelta(minutes=int(ttl_min))
        now = _now_utc()
        orders = api.list_orders(status="open", nested=True)
        for o in orders or []:
            try:
                created = getattr(o, "created_at", None) or getattr(o, "submitted_at", None)
                if isinstance(created, str):
                    try:
                        # Parse RFC3339/ISO format
                        created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    except Exception:
                        created_dt = now
                elif isinstance(created, datetime):
                    created_dt = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
                else:
                    created_dt = now
                if now - created_dt >= ttl:
                    if debug:
                        cid = getattr(o, "id", None) or getattr(o, "client_order_id", None)
                        print(f"[ttl] cancel stale order id={cid} sym={getattr(o,'symbol','')} age={(now-created_dt)}")
                    try:
                        api.cancel_order(getattr(o, "id", None))
                    except Exception:
                        pass
            except Exception:
                continue
    except Exception:
        if debug:
            print("[ttl] error during stale order cancellation")


def main() -> int:
    _load_dotenv_if_present()
    ap = argparse.ArgumentParser(description="Crypto Signals → Alpaca paper trader")
    ap.add_argument("--universe", default=None, help="Universe JSON to read; default: latest in universe_runs/")
    ap.add_argument("--max-coins", type=int, default=6, help="Max coins to include (default: 6)")
    ap.add_argument("--tf", default=os.getenv("TB_TRADER_TF", "1h"), help="Timescale to trade (1h/4h/1D/1W)")
    ap.add_argument("--risk-frac", type=float, default=float(os.getenv("TB_TRADER_RISK_FRAC", "0.005")), help="Fraction of equity to risk per trade (default 0.5%)")
    ap.add_argument("--min-notional", type=float, default=float(os.getenv("TB_TRADER_MIN_NOTIONAL", "10")), help="Minimum order notional in quote currency (default $10)")
    ap.add_argument("--max-notional", type=float, default=float(os.getenv("TB_TRADER_MAX_NOTIONAL", "0")), help="Hard cap on order notional in quote currency (0 disables)")
    ap.add_argument("--symbols", default=os.getenv("TB_TRADER_SYMBOLS", "").strip(), help="Comma-separated symbols to include (e.g., BTC/USD,ETH/USD)")
    ap.add_argument("--loop", action="store_true", help="Run continuously with interval sleeps")
    ap.add_argument("--interval-sec", type=int, default=int(os.getenv("TB_TRADER_INTERVAL_SEC", "60")), help="Loop interval seconds (default 60)")
    ap.add_argument("--cooldown-sec", type=int, default=int(os.getenv("TB_TRADER_COOLDOWN_SEC", "300")), help="Per-symbol cooldown seconds to avoid re-entry (default 300)")
    ap.add_argument("--entry-tolerance-bps", type=float, default=float(os.getenv("TB_TRADER_ENTRY_TOL_BPS", "0")), help="Entry trigger tolerance in basis points (default 0)")
    ap.add_argument("--entry-mid-zone", action="store_true", default=(os.getenv("TB_TRADER_ENTRY_MID_ZONE", "0") == "1"), help="Use mid of entry zone for trigger checks when available")
    ap.add_argument("--order-ttl-min", type=int, default=int(os.getenv("TB_TRADER_ORDER_TTL_MIN", "0")), help="Cancel open orders older than this many minutes (0=disabled)")
    ap.add_argument("--show-current-price", action="store_true", help="When --debug, also print live current price for each candidate")
    ap.add_argument("--min-rr", type=float, default=float(os.getenv("TB_TRADER_MIN_RR", "0.0")), help="Minimum risk-reward ratio required to trade (0 disables)")
    ap.add_argument("--allow-shorts", action="store_true", default=(os.getenv("TB_TRADER_ALLOW_SHORTS", "0") == "1"), help="Allow short sells (default off). When off, SELL requires existing base position > 0")
    ap.add_argument("--longs-only", action="store_true", default=(os.getenv("TB_TRADER_LONGS_ONLY", "0") == "1"), help="Only trade spot longs (BUY). In online mode, allow SELL only when base position > 0; otherwise drop silently. In offline preview, SELLs are suppressed.")
    ap.add_argument("--debug", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    def one_pass() -> int:
        uni = Path(args.universe) if args.universe else _latest_universe_file()
        if not uni or not uni.exists():
            print("[trader] No universe file found. Run universe scanner first.")
            return 1
        try:
            universe = json.loads(Path(uni).read_text())
        except Exception as e:
            print(f"[trader] Failed to read universe: {e}")
            return 1

        digest = _build_digest_data(universe, max_coins=args.max_coins)
        digest.setdefault("provenance", {}).update({"artifact": Path(uni).name, **_provenance()})

        allowed: Optional[Set[str]] = None
        if args.symbols:
            parts = [p.strip().upper() for p in args.symbols.split(",") if p.strip()]
            allowed = { _normalize_symbol(p).upper() for p in parts }

        candidates = _iterate_candidates(digest, args.tf, allowed_symbols=allowed)
        if args.debug:
            print(f"[trader] candidates[{args.tf}]={len(candidates)}")
            for c in candidates:
                # Longs-only: suppress SELL candidate debug lines
                if args.longs_only and c.get('side') == 'sell':
                    continue
                line = f"  - {c['symbol']} {c['side']} entry={c['entry']:.4f} tp={c['tp']:.4f} sl={c['stop']:.4f}"
                if args.show_current_price:
                    px = _current_price(c['symbol'])
                    if px is not None:
                        line += f" | live={px:.4f}"
                print(line)

        offline = _bool("TB_TRADER_OFFLINE", True)
        no_trade = _bool("TB_NO_TRADE", True)
        st = _load_state()

        if offline:
            if args.debug:
                print("[trader] offline=1: no API calls; preview-only.")
            # In offline mode, emit preview intents to journal/discord for observability
            for c in candidates:
                # Longs-only: suppress SELL previews to reduce noise
                if args.longs_only and c.get("side") == "sell":
                    continue
                payload = {
                    "ts": _now_utc().isoformat(),
                    "event": "would_submit",
                    "artifact": Path(uni).name,
                    "symbol": c["symbol"],
                    "tf": args.tf,
                    "side": c["side"],
                    "bias": None,
                    "entry": c["entry"],
                    "stop": c["stop"],
                    "tp": c["tp"],
                    "price": _current_price(c["symbol"]) or "",
                    "qty": "",
                    "risk_frac": args.risk_frac,
                    "cooldown_sec": args.cooldown_sec,
                    "order_id": "",
                    "status": "preview",
                    "note": "offline",
                }
                _journal_append(payload)
                _notify_discord("would_submit", payload, debug=args.debug)
            return 0

        api = _get_alpaca()
        if api is None:
            print("[trader] Alpaca REST unavailable (missing keys or import). Set TB_TRADER_OFFLINE=1 for preview-only.")
            return 2

        try:
            equity = float(getattr(api.get_account(), "equity", 0.0))
        except Exception as e:
            print(f"[trader] failed to read account: {e}")
            return 2

        # Optional: cancel stale open orders before new submissions (disabled in no-trade mode)
        if args.order_ttl_min > 0 and not _bool("TB_NO_TRADE", True):
            _cancel_stale_orders(api, args.order_ttl_min, debug=args.debug)

        ok_all = True
        for c in candidates:
            sym = c["symbol"]
            # Cooldown gate
            if not _cooldown_ok(st, sym, args.cooldown_sec):
                if args.debug:
                    print(f"[gate] cooldown active for {sym}; skip")
                continue
            # Price trigger check (with optional mid-of-zone entry and tolerance)
            px = _current_price(sym)
            if px is None:
                if args.debug:
                    print(f"[gate] no price for {sym}; skip")
                continue
            entry_for_trigger = c["entry"]
            if args.entry_mid_zone:
                mid = _mid_of_zone(c.get("raw_entries"), c["side"])
                if mid is not None:
                    entry_for_trigger = float(mid)
            tol = max(0.0, float(args.entry_tolerance_bps)) / 10000.0
            if c["side"] == "buy":
                trig_ok = px >= entry_for_trigger * (1.0 - tol)
            else:
                trig_ok = px <= entry_for_trigger * (1.0 + tol)
            if not trig_ok:
                if args.debug:
                    print(f"[gate] trigger not met for {sym}: px={px:.4f} vs entry={entry_for_trigger:.4f} tol_bps={args.entry_tolerance_bps}")
                continue
            # Risk-Reward gate (based on plan levels)
            rr = None
            try:
                if c["side"] == "buy":
                    risk = float(entry_for_trigger) - float(c["stop"])
                    reward = float(c["tp"]) - float(entry_for_trigger)
                else:
                    risk = float(c["stop"]) - float(entry_for_trigger)
                    reward = float(entry_for_trigger) - float(c["tp"]) 
                if risk > 0 and reward > 0:
                    rr = reward / risk
                else:
                    rr = 0.0
            except Exception:
                rr = 0.0
            if float(args.min_rr) > 0.0 and (rr is None or rr < float(args.min_rr)):
                if args.debug:
                    print(f"[gate] RR below min for {sym}: rr={rr:.2f} < min_rr={args.min_rr}")
                continue
            # Duplicate protection
            if _has_open_in_direction(api, sym, c["side"]):
                if args.debug:
                    print(f"[gate] open position/order exists for {sym} side={c['side']}; skip")
                continue

            # Position-aware SELL gate (spot): require inventory; shorts unsupported on Alpaca crypto
            if c["side"] == "sell" and not no_trade:
                pos_qty = _position_qty(api, sym)
                if pos_qty <= 0:
                    if not args.allow_shorts or not _broker_supports_crypto_shorts():
                        # In longs-only mode, drop silently to avoid spam; otherwise journal a skip
                        reason = "skipped:no_position_for_sell" if not args.allow_shorts else "skipped:shorts_not_supported"
                        if args.longs_only:
                            if args.debug:
                                print(f"[gate] SELL {sym} suppressed (longs-only, no position)")
                            continue
                        else:
                            if args.debug:
                                print(f"[gate] SELL {sym} blocked: {reason}")
                            payload = {
                                "ts": _now_utc().isoformat(),
                                "event": "skipped",
                                "artifact": Path(uni).name,
                                "symbol": sym,
                                "tf": args.tf,
                                "side": c["side"],
                                "bias": None,
                                "entry": c["entry"],
                                "stop": c["stop"],
                                "tp": c["tp"],
                                "price": px,
                                "qty": "",
                                "risk_frac": args.risk_frac,
                                "cooldown_sec": args.cooldown_sec,
                                "order_id": "",
                                "status": "skipped",
                                "note": reason,
                            }
                            _journal_append(payload)
                            _notify_discord("skipped", payload, debug=args.debug)
                            continue

            qty = _calc_qty(equity, c["entry"], c["stop"], args.risk_frac)
            # Enforce minimum notional for Alpaca paper trading
            min_notional = max(0.0, float(args.min_notional))
            if min_notional > 0 and c["entry"] > 0:
                notional = qty * c["entry"]
                if notional < min_notional:
                    qty = min_notional / c["entry"]
            # Enforce hard maximum notional cap if provided
            max_notional = max(0.0, float(getattr(args, "max_notional", 0.0)))
            if max_notional > 0 and c["entry"] > 0:
                notional2 = qty * c["entry"]
                if notional2 > max_notional:
                    qty = max_notional / c["entry"]
            if qty <= 0:
                if args.debug:
                    print(f"[trade] skip {sym} invalid qty (entry={c['entry']:.4f}, stop={c['stop']:.4f})")
                continue

            # Final SELL safety: cap to available position, or skip if none
            note_extra = ""
            if c["side"] == "sell" and not no_trade:
                pos_qty2 = _position_qty(api, sym)
                if args.debug:
                    print(f"[gate] SELL {sym} pos_qty={pos_qty2:.6f} planned_qty={qty:.6f}")
                if pos_qty2 <= 0:
                    if args.longs_only:
                        if args.debug:
                            print(f"[gate] SELL {sym} suppressed at submit (longs-only, no position)")
                        continue
                    else:
                        if args.debug:
                            print(f"[gate] SELL {sym} blocked at submit: no_position_for_sell")
                        payload = {
                            "ts": _now_utc().isoformat(),
                            "event": "skipped",
                            "artifact": Path(uni).name,
                            "symbol": sym,
                            "tf": args.tf,
                            "side": c["side"],
                            "bias": None,
                            "entry": c["entry"],
                            "stop": c["stop"],
                            "tp": c["tp"],
                            "price": px,
                            "qty": "",
                            "risk_frac": args.risk_frac,
                            "cooldown_sec": args.cooldown_sec,
                            "order_id": "",
                            "status": "skipped",
                            "note": "skipped:no_position_for_sell",
                        }
                        _journal_append(payload)
                        _notify_discord("skipped", payload, debug=args.debug)
                        continue
                if qty > pos_qty2:
                    note_extra = f" qty_capped_to_position({pos_qty2:.6f})"
                    qty = pos_qty2

            if no_trade:
                if args.debug:
                    print(f"[trade] no-trade gate: would submit {sym} {c['side']} qty={qty:.6f}{note_extra}")
                payload = {
                    "ts": _now_utc().isoformat(),
                    "event": "would_submit",
                    "artifact": Path(uni).name,
                    "symbol": sym,
                    "tf": args.tf,
                    "side": c["side"],
                    "bias": None,
                    "entry": c["entry"],
                    "stop": c["stop"],
                    "tp": c["tp"],
                    "price": px,
                    "qty": qty,
                    "risk_frac": args.risk_frac,
                    "cooldown_sec": args.cooldown_sec,
                    "order_id": "",
                    "status": "preview",
                    "note": ((f"no_trade rr={rr:.2f}" if rr is not None else "no_trade") + note_extra).strip(),
                }
                _journal_append(payload)
                _notify_discord("would_submit", payload, debug=args.debug)
                _mark_action(st, sym)
                continue

            ok, oid, err = _place_bracket(api, sym, c["side"], qty, c["entry"], c["tp"], c["stop"], debug=args.debug)
            payload = {
                "ts": _now_utc().isoformat(),
                "event": "submit",
                "artifact": Path(uni).name,
                "symbol": sym,
                "tf": args.tf,
                "side": c["side"],
                "bias": None,
                "entry": c["entry"],
                "stop": c["stop"],
                "tp": c["tp"],
                "price": px,
                "qty": qty,
                "risk_frac": args.risk_frac,
                "cooldown_sec": args.cooldown_sec,
                "order_id": oid or "",
                "status": "submitted" if ok else "failed",
                "note": ((f"rr={rr:.2f}" if rr is not None else "") + note_extra + (f" err={err}" if (not ok and err) else "")).strip(),
            }
            _journal_append(payload)
            _notify_discord("submit", payload, debug=args.debug)
            if ok:
                _mark_action(st, sym)
            ok_all = ok_all and ok
            time.sleep(0.25)

        _save_state(st)
        return 0 if ok_all else 3

    if args.loop:
        if args.debug:
            print(f"[loop] starting with interval={args.interval_sec}s; tf={args.tf}")
        while True:
            rc = one_pass()
            time.sleep(max(5, int(args.interval_sec)))
        # unreachable
    else:
        return one_pass()


if __name__ == "__main__":
    raise SystemExit(main())
