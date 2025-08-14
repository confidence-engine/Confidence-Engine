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
from scripts.discord_sender import send_discord_digest
from scripts.discord_formatter import digest_to_discord_embeds

# Lightweight .env loader to ensure PPLX/TB_* envs are available when running directly
def _load_dotenv_if_present():
    root = Path(__file__).parent.parent
    env_path = root / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if '=' not in s:
                continue
            key, val = s.split('=', 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # Do not override pre-set environment variables
            if key and (key not in os.environ or os.environ.get(key, '') == ''):
                os.environ[key] = val
    except Exception:
        pass

_load_dotenv_if_present()
try:
    from scripts.polymarket_bridge import discover_from_env as discover_polymarket
except Exception:
    def discover_polymarket(*args, **kwargs):
        return []
from alpaca import recent_bars as alpaca_recent_bars

# Public helper to enrich a saved universe artifact with per-asset evidence lines
# and a top-level polymarket array. Pure storage change; chat outputs unchanged.
from typing import Any
def enrich_artifact(universe_file: str, ev_sink: Dict[str, str], poly_items: List[Dict[str, Any]], assets_data: Dict[str, dict] = None):
    if not universe_file:
        return
    import json
    data: Dict[str, Any] = {}
    try:
        with open(universe_file, "r") as f:
            data = json.load(f)
    except Exception:
        return
    payloads = data.get("payloads") or []
    # Attach per-payload evidence_line; None if absent/disabled
    ev_map = ev_sink or {}
    for p in payloads:
        sym = p.get("symbol")
        p["evidence_line"] = ev_map.get(sym) if sym in ev_map else None
        # Persist key thesis fields and plan snapshot when available
        try:
            ainfo = (assets_data or {}).get(sym, {}) if assets_data else {}
            th_src = (ainfo.get("thesis") or {}) if isinstance(ainfo, dict) else {}
            if th_src:
                th_dst = (p.get("thesis") or {}) if isinstance(p.get("thesis"), dict) else {}
                # Only persist friendly fields used by chat
                for k in ("action", "risk_band", "readiness"):
                    v = th_src.get(k)
                    if v is not None:
                        th_dst[k] = v
                if th_dst:
                    p["thesis"] = th_dst
            # Plan snapshot (per-TF entries/invalid/targets)
            plan_src = ainfo.get("plan") if isinstance(ainfo, dict) else None
            if isinstance(plan_src, dict) and plan_src:
                p["plan"] = plan_src
        except Exception:
            pass
    # Build top-level polymarket array from already filtered/mapped items
    poly_out = []
    for it in (poly_items or []):
        try:
            poly_out.append({
                "market_name": it.get("title"),
                "stance": it.get("stance"),
                "readiness": it.get("readiness"),
                "edge_label": it.get("edge_label"),
                "rationale": it.get("rationale_chat"),
                "implied_prob": it.get("implied_prob"),
                "implied_pct": it.get("implied_pct"),
                "tb_internal_prob": it.get("internal_prob"),
                "liquidity_usd": it.get("liquidity_usd"),
                "event_end_date": it.get("end_date_iso"),
                "market_id": it.get("market_id"),
                "quality_score": it.get("quality"),
            })
        except Exception:
            continue
    data["polymarket"] = poly_out
    try:
        with open(universe_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# Public helper for tests: select BTC/ETH + top-K alts (payloads already ranked)
def select_digest_symbols_public(payloads_list: List[Dict], k_alts: int) -> List[str]:
    b = [p for p in payloads_list if (p.get("symbol", "").upper().startswith("BTC/"))]
    e = [p for p in payloads_list if (p.get("symbol", "").upper().startswith("ETH/"))]
    others = [p for p in payloads_list if (p.get("symbol_type") == "crypto") and p not in b + e]
    selected = b + e + others[: max(0, int(k_alts))]
    return [p.get("symbol") for p in selected if p.get("symbol")]

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
        other_crypto_all = [p for p in payloads if is_crypto(p) and p not in btc + eth]
        stocks = [p for p in payloads if not is_crypto(p)]
        ordered_payloads = (btc + eth + other_crypto_all + stocks)
        assets_ordered_all: List[str] = [p.get("symbol") for p in ordered_payloads if p.get("symbol")]

        # Select digest-surfaced symbols: BTC/ETH + top-K alts (no stocks)
        _top_env = os.getenv("TB_DIGEST_TOP_ALTS", os.getenv("TB_DIGEST_MAX_ALTS", "5"))
        try:
            if isinstance(_top_env, str) and _top_env.strip().lower() in ("all", "-1"):
                top_k_alts = len(other_crypto_all)
            else:
                top_k_alts = int(_top_env)
        except Exception:
            top_k_alts = 5
        assets_ordered_digest: List[str] = select_digest_symbols_public(ordered_payloads, top_k_alts)

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

        def get_stock_spot_price(symbol: str):
            # Stocks: Alpaca primary, PPLX fallback (no Binance)
            try:
                price = alpaca_spot_price(symbol)
                if price:
                    return price
            except Exception as e:
                print(f"[WARN] Alpaca stock price fetch failed for {symbol}: {e}")
            try:
                price = pplx_spot_price(symbol)
                if price:
                    return price
            except Exception as e:
                print(f"[WARN] PPLX stock price fetch failed for {symbol}: {e}")
            return None

        # ---- Crypto TF plan helpers (do NOT change rotation helpers above) ----
        ORDERED_TFS = ["1h", "4h", "1D", "1W", "1M"]

        def build_tf_plan_from_levels(levels: dict):
            if not levels:
                return None
            entries = []
            trg = levels.get("entry_trigger")
            zone = levels.get("entry_zone")
            e_type = levels.get("entry_type") or ("trigger" if isinstance(trg, (int, float)) else "fade")

            if isinstance(trg, (int, float)):
                entries.append({"type": e_type, "zone_or_trigger": float(trg)})
            elif isinstance(zone, (list, tuple)) and len(zone) == 2 and all(isinstance(x, (int, float)) for x in zone):
                lo, hi = zone
                entries.append({"type": e_type, "zone_or_trigger": [float(lo), float(hi)]})

            invalidation = None
            if isinstance(levels.get("invalid_price"), (int, float)):
                invalidation = {
                    "price": float(levels["invalid_price"]),
                    "condition": levels.get("invalid_condition", "close below"),
                }

            targets = []
            for i, tp in enumerate(levels.get("targets") or [], start=1):
                if isinstance(tp, (int, float)):
                    targets.append({"label": f"TP{i}", "price": float(tp)})
                elif isinstance(tp, dict) and isinstance(tp.get("price"), (int, float)):
                    targets.append({"label": tp.get("label") or f"TP{i}", "price": float(tp["price"])})

            if not entries and not invalidation and not targets:
                return None
            return {"entries": entries, "invalidation": invalidation, "targets": targets}

        def collect_tf_levels(symbol: str, tf: str, analysis_map: dict):
            per_tf = (analysis_map.get("levels") or {}).get(tf)
            return build_tf_plan_from_levels(per_tf)

        def build_plan_all_tfs(symbol: str, analysis_map: dict):
            plan = {}
            for tf in ORDERED_TFS:
                tf_plan = collect_tf_levels(symbol, tf, analysis_map)
                if tf_plan:
                    # Tag provenance: analysis-derived
                    tf_plan["source"] = "analysis"
                    plan[tf] = tf_plan
            return plan

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
            # Compute spot via rotations for crypto only; skip stocks to avoid provider noise
            spot_price = get_crypto_spot_price(sym) if is_crypto else None

            # Build plans
            if is_crypto:
                # Prefer existing analysis-derived levels per TF
                base_plan = build_plan_all_tfs(sym, p)
                if not base_plan:
                    # Fallback: minimal heuristic across TFs if we have a spot
                    base_plan = {}
                    if spot_price:
                        # TF-specific percentage offsets (heuristic) to avoid identical levels across TFs
                        tf_pcts = {
                            "1h":   {"trig": 0.003, "fade_hi": 0.007, "fade_lo": 0.004, "inv": 0.008, "tp1": 0.007, "tp2": 0.015},
                            "4h":   {"trig": 0.005, "fade_hi": 0.010, "fade_lo": 0.006, "inv": 0.015, "tp1": 0.012, "tp2": 0.025},
                            "1D":   {"trig": 0.008, "fade_hi": 0.015, "fade_lo": 0.010, "inv": 0.020, "tp1": 0.020, "tp2": 0.040},
                            "1W":   {"trig": 0.015, "fade_hi": 0.030, "fade_lo": 0.020, "inv": 0.035, "tp1": 0.040, "tp2": 0.080},
                            "1M":   {"trig": 0.030, "fade_hi": 0.060, "fade_lo": 0.040, "inv": 0.060, "tp1": 0.080, "tp2": 0.150},
                            "daily": {"trig": 0.008, "fade_hi": 0.015, "fade_lo": 0.010, "inv": 0.020, "tp1": 0.020, "tp2": 0.040},
                        }
                        for tf in ORDERED_TFS:
                            pcts = tf_pcts.get(str(tf), tf_pcts.get("daily"))
                            if thesis["bias"] == "up":
                                entries_items = [{"type": "trigger", "zone_or_trigger": spot_price * (1.0 + pcts["trig"])}]
                                invalid = {"price": spot_price * (1.0 - pcts["inv"]), "condition": "close below"}
                                targets = [
                                    {"label": "TP1", "price": spot_price * (1.0 + pcts["tp1"])},
                                    {"label": "TP2", "price": spot_price * (1.0 + pcts["tp2"])},
                                ]
                            elif thesis["bias"] == "down":
                                entries_items = [{"type": "fade", "zone_or_trigger": [spot_price * (1.0 - pcts["fade_hi"]), spot_price * (1.0 - pcts["fade_lo"]) ]}]
                                invalid = {"price": spot_price * (1.0 + pcts["inv"]), "condition": "close above"}
                                targets = [
                                    {"label": "TP1", "price": spot_price * (1.0 - pcts["tp1"])},
                                    {"label": "TP2", "price": spot_price * (1.0 - pcts["tp2"])},
                                ]
                            else:
                                entries_items = [{"type": "trigger", "zone_or_trigger": spot_price}]
                                invalid = {"price": spot_price * (1.0 - max(0.01, pcts["trig"])), "condition": "breach"}
                                targets = [{"label": "TP1", "price": spot_price * (1.0 + max(0.01, pcts["trig"]))}]
                            base_plan[tf] = {"entries": entries_items, "invalidation": invalid, "targets": targets, "source": "fallback"}
                    else:
                        base_plan = {}
            else:
                # Stocks: do NOT auto-populate TF plans; only provide spot and preserve any existing plan if present
                base_plan = p.get("plan") or {}
            assets_data[sym] = {
                "spot": spot_price,
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
        # Provenance (artifact file + git sha) computed once for both TG/Discord
        provenance = {}
        try:
            art_name = os.path.basename(summary.get("universe_file") or "")
            if art_name:
                provenance["artifact"] = art_name
        except Exception:
            pass
        try:
            import subprocess
            r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
            sha = (r.stdout or "").strip()
            if sha:
                provenance["git"] = sha
        except Exception:
            pass
        options = {
            "include_weekly": os.getenv("TB_DIGEST_INCLUDE_WEEKLY", "1") == "1",
            "include_engine": os.getenv("TB_DIGEST_INCLUDE_ENGINE", "1") == "1",
            "max_tfs": int(os.getenv("TB_DIGEST_MAX_TFS", "2")),
            "drift_warn_pct": float(os.getenv("TB_DIGEST_DRIFT_WARN_PCT", "0.5")),
            "include_prices": os.getenv("TB_DIGEST_INCLUDE_PRICES", "1") == "1",
            "provenance": provenance,
        }
        # Polymarket (optional)
        polymarket_items = []
        try:
            # Build richer context for BTC/ETH from current signals
            def _first_by_prefix(d: Dict[str, dict], pref: str) -> dict:
                for k, v in d.items():
                    if isinstance(k, str) and k.upper().startswith(pref):
                        return v or {}
                return {}

            def _derive_align_score(info: dict) -> float:
                # Use timescale_scores and bias to estimate alignment in [0,1]
                th = (info.get("thesis") or {})
                bias = str(th.get("bias") or "").lower()
                tfs = info.get("timescale_scores") or {}
                if not isinstance(tfs, dict) or not tfs:
                    return 0.5
                vals = []
                if bias == "up":
                    for v in tfs.values():
                        try:
                            vals.append(1.0 if float(v) > 0 else 0.0)
                        except Exception:
                            pass
                elif bias == "down":
                    for v in tfs.values():
                        try:
                            vals.append(1.0 if float(v) < 0 else 0.0)
                        except Exception:
                            pass
                else:
                    for v in tfs.values():
                        try:
                            x = float(v)
                            vals.append(1.0 - abs(x))  # neutrality favors alignment when bias neutral
                        except Exception:
                            pass
                if not vals:
                    return 0.5
                return max(0.0, min(1.0, sum(vals) / len(vals)))

            def _ctx_from(info: dict) -> dict:
                th = (info.get("thesis") or {})
                return {
                    "action": th.get("action") or info.get("action"),
                    "readiness": th.get("readiness") or info.get("readiness"),
                    "risk_band": th.get("risk_band") or info.get("risk_band"),
                    "align_score": _derive_align_score(info),
                }

            btc_info = _first_by_prefix(assets_data, "BTC/")
            eth_info = _first_by_prefix(assets_data, "ETH/")
            ctx = {
                "BTC": _ctx_from(btc_info),
                "ETH": _ctx_from(eth_info),
            }
            polymarket_items = discover_polymarket(context=ctx)
        except Exception:
            polymarket_items = []
        evidence_sink: Dict[str, str] = {}
        text = tg_fmt.render_digest(
            timestamp_utc=summary.get("timestamp_iso") or "",
            weekly=weekly,
            engine=engine,
            assets_ordered=assets_ordered_digest,
            assets_data=assets_data,
            options=options,
            polymarket=polymarket_items,
        )
        print("\nTelegram Digest (Human):")
        print("-----------------------")

        # Enrich saved artifact with evidence lines and polymarket array (storage only; no chat change)
        try:
            ufile = summary.get("universe_file")
            enrich_artifact(ufile, evidence_sink, polymarket_items, assets_data=assets_data)
            # After enrichment, the universe file is modified post scan_universe's commit.
            # If auto-commit is enabled, commit and optionally push the enrichment changes now.
            try:
                autoc = os.getenv("TB_UNIVERSE_GIT_AUTOCOMMIT", "1") == "1"
                pushc = os.getenv("TB_UNIVERSE_GIT_PUSH", "1") == "1"
                if autoc and ufile:
                    import subprocess
                    # Stage the modified universe file and metrics.csv if present
                    subprocess.run(["git", "add", ufile, "universe_runs/metrics.csv"], check=False)
                    msg = f"universe: enrichment (evidence_line + polymarket) for {os.path.basename(ufile)}"
                    # Commit only if there is something to commit
                    r = subprocess.run(["git", "commit", "-m", msg], check=False)
                    if r.returncode == 0:
                        print("[Universe] Enrichment auto-commit done.")
                        if pushc:
                            try:
                                subprocess.run(["git", "push"], check=True)
                                print("[Universe] Enrichment pushed.")
                            except Exception as pe:
                                print(f"[Universe] Enrichment push failed: {pe}")
                    else:
                        print("[Universe] Enrichment commit skipped (no changes).")
            except Exception as ge:
                print(f"[Universe] Enrichment auto-commit skipped: {ge}")
        except Exception as e:
            print(f"[Artifact] Enrichment skipped: {e}")
        print(text)
        print("-----------------------")

        # Unified send gating: try to send to both if creds/flags allow
        HUMAN_DIGEST = os.getenv("TB_HUMAN_DIGEST", "1") == "1"
        SEND_TG = (
            os.getenv("TB_NO_TELEGRAM", "0") == "0"
            and os.getenv("TELEGRAM_BOT_TOKEN")
            and os.getenv("TELEGRAM_CHAT_ID")
            and (not args.no_telegram)
        )
        SEND_DISCORD = (
            os.getenv("TB_NO_DISCORD", "0") == "0"
            and os.getenv("TB_ENABLE_DISCORD", "1") == "1"
            and os.getenv("DISCORD_WEBHOOK_URL")
        )

        if HUMAN_DIGEST:
            # Build structured digest_data once
            exec_take = ""
            if isinstance(engine, str) and engine:
                for ln in engine.splitlines():
                    if ln.strip():
                        exec_take = ln.strip()
                        break
            if not exec_take and isinstance(weekly, str) and weekly:
                for ln in weekly.splitlines():
                    if ln.strip():
                        exec_take = ln.strip()
                        break

            assets_list = []
            for sym in assets_ordered_digest:
                a = assets_data.get(sym, {})
                th = a.get("thesis") or {}
                assets_list.append({
                    "symbol": sym,
                    "spot": a.get("spot"),
                    "structure": a.get("structure"),
                    "risk": th.get("risk_band") or "Medium",
                    "readiness": th.get("readiness") or "Later",
                    "action": th.get("action") or "Watch",
                    "plan": a.get("plan") or {},
                })

            digest_data = {
                "timestamp": summary.get("timestamp_iso") or "",
                "executive_take": exec_take,
                "weekly": weekly or "",
                "engine": engine or "",
                "assets": assets_list,
                "polymarket": polymarket_items or [],
                "provenance": provenance,
            }

            if SEND_TG:
                try:
                    from scripts.tg_sender import send_telegram_text_multi
                    sent = send_telegram_text_multi(text)
                    print(f"[TG] Sent (multi): {bool(sent)}")
                except Exception as e:
                    print(f"[TG] Error: {e}")
            else:
                print("[TG] Skipped — missing credentials or disabled")

            if SEND_DISCORD:
                try:
                    embeds = digest_to_discord_embeds(digest_data)
                    d_ok = send_discord_digest(embeds)
                    print(f"[Discord] Sent: {bool(d_ok)}")
                except Exception as e:
                    print(f"[Discord] Error: {e}")
            else:
                print("[Discord] Skipped — missing webhook or disabled")

if __name__ == "__main__":
    main()
