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
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

# Project path
_THIS_DIR = Path(__file__).parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# Local imports
from scripts.crypto_signals_digest import _load_dotenv_if_present, _latest_universe_file, _build_digest_data  # type: ignore
from alpaca import recent_bars as alpaca_recent_bars  # price helper

try:
    from alpaca_trade_api.rest import REST, TimeInForce, OrderSide, OrderType
except Exception:  # pragma: no cover
    REST = None  # type: ignore
    TimeInForce = None  # type: ignore
    OrderSide = None  # type: ignore
    OrderType = None  # type: ignore


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
    if not key or not sec:
        return None
    try:
        return REST(key_id=key, secret_key=sec, base_url=url)
    except Exception:
        return None


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


def _decimals_for(sym: str) -> int:
    # Simple heuristic for crypto prices; refine per-symbol later
    base = (sym or "").split("/")[0]
    if base in ("BTC", "ETH"):
        return 2
    return 4


def _place_bracket(api: REST, symbol: str, side: str, qty: float, entry: float, tp: float, sl: float, debug: bool = False) -> Tuple[bool, Optional[str]]:
    try:
        tif = TimeInForce.gtc if hasattr(TimeInForce, "gtc") else "gtc"
        otype = OrderType.market if hasattr(OrderType, "market") else "market"
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
        return True, str(oid) if oid else None
    except Exception as e:
        if debug:
            print(f"[trade] submit failed: {e}")
        return False, None


def _decide_side(thesis: Dict[str, Any]) -> Optional[str]:
    bias = str(thesis.get("bias", "")).lower()
    if bias == "up":
        return "buy"
    if bias == "down":
        return "sell"
    return None


def _iterate_candidates(digest: Dict[str, Any], tf: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a in (digest.get("assets") or []):
        sym = a.get("symbol")
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
        tp1 = targets[0].get("price") if isinstance(targets[0], dict) else None
        if tp1 is None:
            continue
        side = _decide_side(a.get("thesis") or {})
        if not side:
            continue
        # entries might be float or [lo, hi]
        if isinstance(entries, (int, float)):
            entry_price = float(entries)
        elif isinstance(entries, (list, tuple)) and len(entries) == 2:
            lo, hi = entries
            entry_price = float(lo if side == "sell" else hi)
        else:
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
    try:
        df = alpaca_recent_bars(symbol, minutes=5)
        if df is not None and len(df) > 0:
            return float(df["close"].iloc[-1])
    except Exception:
        return None
    return None


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


def main() -> int:
    _load_dotenv_if_present()
    ap = argparse.ArgumentParser(description="Crypto Signals → Alpaca paper trader")
    ap.add_argument("--universe", default=None, help="Universe JSON to read; default: latest in universe_runs/")
    ap.add_argument("--max-coins", type=int, default=6, help="Max coins to include (default: 6)")
    ap.add_argument("--tf", default=os.getenv("TB_TRADER_TF", "1h"), help="Timescale to trade (1h/4h/1D/1W)")
    ap.add_argument("--risk-frac", type=float, default=float(os.getenv("TB_TRADER_RISK_FRAC", "0.005")), help="Fraction of equity to risk per trade (default 0.5%)")
    ap.add_argument("--loop", action="store_true", help="Run continuously with interval sleeps")
    ap.add_argument("--interval-sec", type=int, default=int(os.getenv("TB_TRADER_INTERVAL_SEC", "60")), help="Loop interval seconds (default 60)")
    ap.add_argument("--cooldown-sec", type=int, default=int(os.getenv("TB_TRADER_COOLDOWN_SEC", "900")), help="Per-symbol cooldown seconds to avoid re-entry (default 900)")
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

        candidates = _iterate_candidates(digest, args.tf)
        if args.debug:
            print(f"[trader] candidates[{args.tf}]={len(candidates)}")
            for c in candidates:
                print(f"  - {c['symbol']} {c['side']} entry={c['entry']:.4f} tp={c['tp']:.4f} sl={c['stop']:.4f}")

        offline = _bool("TB_TRADER_OFFLINE", True)
        no_trade = _bool("TB_NO_TRADE", True)
        st = _load_state()

        if offline:
            if args.debug:
                print("[trader] offline=1: no API calls; preview-only.")
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

        ok_all = True
        for c in candidates:
            sym = c["symbol"]
            # Cooldown gate
            if not _cooldown_ok(st, sym, args.cooldown_sec):
                if args.debug:
                    print(f"[gate] cooldown active for {sym}; skip")
                continue
            # Price trigger check
            px = _current_price(sym)
            if px is None:
                if args.debug:
                    print(f"[gate] no price for {sym}; skip")
                continue
            trig_ok = (px >= c["entry"]) if c["side"] == "buy" else (px <= c["entry"])
            if not trig_ok:
                if args.debug:
                    print(f"[gate] trigger not met for {sym}: px={px:.4f} vs entry={c['entry']:.4f}")
                continue
            # Duplicate protection
            if _has_open_in_direction(api, sym, c["side"]):
                if args.debug:
                    print(f"[gate] open position/order exists for {sym} side={c['side']}; skip")
                continue

            qty = _calc_qty(equity, c["entry"], c["stop"], args.risk_frac)
            if qty <= 0:
                if args.debug:
                    print(f"[trade] skip {sym} invalid qty (entry={c['entry']:.4f}, stop={c['stop']:.4f})")
                continue

            if no_trade:
                if args.debug:
                    print(f"[trade] no-trade gate: would submit {sym} {c['side']} qty={qty:.6f}")
                _mark_action(st, sym)
                continue

            ok, _ = _place_bracket(api, sym, c["side"], qty, c["entry"], c["tp"], c["stop"], debug=args.debug)
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
