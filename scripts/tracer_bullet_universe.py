#!/usr/bin/env python3
"""
Canonical multi-asset universe entrypoint for Tracer Bullet.
"""
import sys
import os
import argparse
import requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.scan_universe import run_universe_scan
from typing import Dict, List

# New additive imports for human digest
from scripts import tg_weekly_engine
from scripts import tg_digest_formatter as tg_fmt
from scripts.tg_sender import send_telegram_text
from alpaca import recent_bars as alpaca_recent_bars

def main():
    parser = argparse.ArgumentParser(description="Tracer Bullet Universe Scanner")
    parser.add_argument("--config", default="config/universe.yaml", help="Universe config file")
    parser.add_argument("--symbols", help="Override symbols (comma-separated)")
    parser.add_argument("--top", type=int, default=5, help="Number of top signals to show")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram sends")
    parser.add_argument("--version-tag", default="v3.1", help="Version tag for this run")
    parser.add_argument("--fail-fast", action="store_true", help="Fail on first error (CI)")
    parser.add_argument("--max-symbols", type=int, default=None, help="Max symbols to scan")
    parser.add_argument("--no-human-digest", action="store_true", help="Disable human-readable digest")
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    
    # Set human digest environment variable based on CLI flag
    if args.no_human_digest:
        os.environ["TB_HUMAN_DIGEST"] = "0"
    # Universe scan: disable internal Telegram sends to avoid double-send.
    summary = run_universe_scan(
        config_path=args.config,
        symbols=symbols,
        top_k=args.top,
        debug=args.debug,
        no_telegram=True,
        version_tag=args.version_tag,
        fail_fast=args.fail_fast,
        max_symbols=args.max_symbols
    )
    print(f"[{args.version_tag}] [TracerBullet Universe] ts={summary.get('timestamp_iso')} top={summary['top_k']}")
    print(f"[{summary.get('version','v3.1')}] Universe scanned: {summary['total_symbols']} symbols; Top {summary['top_k']} ranked. File: {summary['universe_file']}")
    # Print technical digest summary if available
    if summary.get("digest") and not summary.get("human_digest"):
        print("\nTop-N Digest:")
        print("\n".join(summary["digest"].splitlines()[:args.top+2]))  # header + N lines

    # Build human Telegram digest (Weekly + Engine + Asset blocks) if enabled
    human_enabled = os.getenv("TB_HUMAN_DIGEST", "1") != "0"
    payloads: List[Dict] = summary.get("payloads") or []
    if human_enabled and payloads:
        # Order: BTC, ETH, other crypto, then stocks
        def is_crypto(p):
            return (p.get("symbol_type") == "crypto")
        btc = [p for p in payloads if p.get("symbol", "").upper().startswith("BTC/")]
        eth = [p for p in payloads if p.get("symbol", "").upper().startswith("ETH/")]
        other_crypto = [p for p in payloads if is_crypto(p) and p not in btc + eth]
        stocks = [p for p in payloads if not is_crypto(p)]
        ordered_payloads = (btc + eth + other_crypto + stocks)
        assets_ordered: List[str] = [p.get("symbol") for p in ordered_payloads if p.get("symbol")]

        # Map payloads to assets_data with minimal fields available
        assets_data: Dict[str, dict] = {}
        # Helper: live spot for crypto via Binance public API (primary)
        def _map_to_binance(sym: str) -> str:
            s = (sym or "").upper()
            base, _, quote = s.partition("/")
            q = quote or "USDT"
            if q in ("USD", "USDC"):
                q = "USDT"
            return f"{base}{q}"

        def binance_spot_price(sym: str):
            try:
                bsym = _map_to_binance(sym)
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={bsym}"
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    return float(r.json().get("price"))
            except Exception as e:
                # Defer handling to rotation wrapper
                raise e

        def _normalize_crypto_symbol(sym: str) -> str:
            # Convert BTCUSD -> BTC/USD; preserve BTC/USD as-is
            if "/" in sym:
                return sym
            if len(sym) >= 6:
                base = sym[:-3]
                quote = sym[-3:]
                return f"{base}/{quote}"
            return sym

        def alpaca_spot_price(sym: str):
            # Use existing alpaca recent_bars to get the last close
            try:
                norm = _normalize_crypto_symbol(sym)
                df = alpaca_recent_bars(norm, minutes=5)
                if df is not None and len(df) > 0 and "close" in df.columns:
                    return float(df["close"].iloc[-1])
            except Exception as e:
                raise e
            return None

        def pplx_spot_price(sym: str):
            # Placeholder: no direct price fetcher available; return None
            # Hook for future Perplexity Pro price provider integration
            return None

        def get_crypto_spot_price(symbol: str):
            # Try Binance first
            try:
                price = binance_spot_price(symbol)
                if price:
                    return price
            except Exception as e:
                print(f"[WARN] Binance price fetch failed for {symbol}: {e}")
            # Fallback to Alpaca
            try:
                price = alpaca_spot_price(symbol)
                if price:
                    return price
            except Exception as e:
                print(f"[WARN] Alpaca price fetch failed for {symbol}: {e}")
            # Fallback to Perplexity Pro (placeholder)
            try:
                price = pplx_spot_price(symbol)
                if price:
                    return price
            except Exception as e:
                print(f"[WARN] PPLX price fetch failed for {symbol}: {e}")
            return None

        for p in ordered_payloads:
            sym = p.get("symbol")
            if not sym:
                continue
            thesis = {
                "bias": ("up" if (p.get("divergence", 0) or 0) > 0 else ("down" if (p.get("divergence", 0) or 0) < 0 else "neutral")),
                "readiness": ("Now" if (p.get("confidence", 0) or 0) > 0.75 else ("Near" if (p.get("confidence", 0) or 0) > 0.65 else "Later")),
                "risk_band": ("High" if (p.get("position_sizing", {}).get("target_R", 0) or 0) > 0.6 else ("Medium" if (p.get("position_sizing", {}).get("target_R", 0) or 0) > 0.3 else "Low")),
                "action": ("Buy" if (p.get("divergence", 0) > 0 and p.get("confidence", 0) > 0.7) else ("Sell" if (p.get("divergence", 0) < 0 and p.get("confidence", 0) > 0.7) else "Watch")),
            }
            is_crypto = (p.get("symbol_type") == "crypto")
            # Plans per timeframe
            base_plan = {"1h": {}, "4h": {}, "daily": {}}
            spot_price = get_crypto_spot_price(sym) if is_crypto else None
            if is_crypto and spot_price:
                # Simple numeric heuristics per bias
                if thesis["bias"] == "up":
                    entries = spot_price * 1.005  # breakout trigger
                    invalid = {"price": spot_price * 0.975, "condition": "close below"}
                    targets = [
                        {"price": spot_price * 1.02},
                        {"price": spot_price * 1.04},
                    ]
                elif thesis["bias"] == "down":
                    entries = [spot_price * 0.995, spot_price * 0.985]  # fade into supply lower-highs
                    invalid = {"price": spot_price * 1.015, "condition": "close above"}
                    targets = [
                        {"price": spot_price * 0.98},
                        {"price": spot_price * 0.96},
                    ]
                else:
                    entries = spot_price  # wait/trigger around spot
                    invalid = {"price": spot_price * 0.98, "condition": "breach"}
                    targets = [{"price": spot_price * 1.01}]
                for tf in base_plan.keys():
                    base_plan[tf] = {
                        "entries": entries,
                        "invalidation": invalid,
                        "targets": targets,
                    }
            else:
                # Narrative scaffolding (no numbers)
                if is_crypto:
                    entry_text = "breakout trigger" if thesis["bias"] == "up" else ("fade into supply" if thesis["bias"] == "down" else "wait for trigger")
                    targets_text = "trail into strength" if thesis["bias"] == "up" else ("cover into weakness" if thesis["bias"] == "down" else "define profit path")
                    for tf in base_plan.keys():
                        base_plan[tf] = {
                            "entries": entry_text,
                            "invalidation": None,
                            "targets": [targets_text],
                        }
            assets_data[sym] = {
                "spot": (spot_price if is_crypto else None),
                "thesis": thesis,
                "structure": tg_fmt.__name__ and "trend / range context",
                "plan": base_plan,
                "weekly_anchor": {},
                "drift_since_snapshot_pct": None,
                "sizing_text": "size by conviction and volatility",
                "timescale_scores": p.get("timescale_scores", {}),
            }

        # Build sections
        weekly = tg_weekly_engine.build_weekly_overview(assets_data)
        engine = tg_weekly_engine.build_engine_minute(assets_data)
        options = {
            "include_weekly": os.getenv("TB_DIGEST_INCLUDE_WEEKLY", "1") == "1",
            "include_engine": os.getenv("TB_DIGEST_INCLUDE_ENGINE", "1") == "1",
            "max_tfs": int(os.getenv("TB_DIGEST_MAX_TFS", "2")),
            "drift_warn_pct": float(os.getenv("TB_DIGEST_DRIFT_WARN_PCT", "0.5")),
            "include_prices": os.getenv("TB_DIGEST_INCLUDE_PRICES", "1") == "1",
        }
        text = tg_fmt.render_digest(
            timestamp_utc=summary.get("timestamp_iso") or "",
            weekly=weekly,
            engine=engine,
            assets_ordered=assets_ordered,
            assets_data=assets_data,
            options=options,
        )
        print("\nTelegram Digest (Human):")
        print("-----------------------")
        print(text)
        print("-----------------------")

        # Conditional Telegram send
        if os.getenv("TB_HUMAN_DIGEST", "1") == "1":
            sent = send_telegram_text(text) if not args.no_telegram else False
            print(f"[Telegram] Human digest sent: {bool(sent)} (skip={args.no_telegram or os.getenv('TB_NO_TELEGRAM','0')=='1'})")

if __name__ == "__main__":
    main()
