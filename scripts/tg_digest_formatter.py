#!/usr/bin/env python3
"""
Telegram human digest formatter (plain text, narrative-first).
Adds Weekly Overview and Engine in One Minute sections and per-asset blocks.
"""
from typing import Dict, List
import os
try:
    # When imported as package
    from .evidence_lines import generate_evidence_line
except Exception:
    # When executed directly
    from scripts.evidence_lines import generate_evidence_line

CRYPTO_PREFIXES = (
    "BTC/", "ETH/", "SOL/", "ADA/", "BNB/", "XRP/", "DOGE/", "DOT/", "LTC/", "AVAX/",
)


def is_crypto(sym: str) -> bool:
    if not sym:
        return False
    s = sym.upper()
    if any(stable in s for stable in ("/USD", "/USDT", "/USDC")):
        return True
    return s.startswith(CRYPTO_PREFIXES)


def _hdr_val(v: str, fallback: str) -> str:
    return v if v else fallback


def render_digest(
    timestamp_utc: str,
    weekly: Dict,
    engine: Dict,
    assets_ordered: List[str],
    assets_data: Dict[str, dict],
    options: Dict[str, str],
) -> str:
    lines: List[str] = []

    # Title
    lines.append(f"Tracer Bullet — {timestamp_utc}")
    lines.append("")

    # Executive Take
    thesis = (engine or {}).get("thesis_text") or "Market mixed; trade clean plans and defined risk."
    lines.append("Executive Take")
    lines.append(thesis)
    lines.append("")

    # Weekly Overview
    if options.get("include_weekly", True):
        lines.append("Weekly Overview")
        lines.append(f"Regime: {weekly.get('regime', 'mixed')}")
        anchors = weekly.get("anchors", [])[:2]
        if anchors:
            parts = []
            for a in anchors:
                t = a.get("type", "zone")
                z = a.get("zone", [])
                # Numeric weekly levels if available
                if isinstance(z, (list, tuple)) and len(z) == 2 and all(isinstance(x, (int, float)) for x in z):
                    lo, hi = z
                    label = t if t in ("supply", "demand") else "zone"
                    parts.append(f"{label} {lo:.0f}–{hi:.0f}")
                else:
                    if t == "supply":
                        parts.append("supply L–H")
                    elif t == "demand":
                        parts.append("demand L–H")
                    else:
                        parts.append("zone L–H")
            lines.append("Weekly Levels: " + "; ".join(parts))
        plan_text = weekly.get("plan_text")
        if plan_text:
            lines.append("Weekly Plan: " + plan_text)
        cats = weekly.get("catalysts", [])
        if cats:
            lines.append("Catalysts: " + "; ".join(cats))
        lines.append("")

    # Engine in One Minute
    if options.get("include_engine", True):
        lines.append("Engine in One Minute")
        lines.append("Thesis: " + ((engine or {}).get("thesis_text") or "Clear triggers over noise."))
        ev = (engine or {}).get("evidence_bullets", [])[:2]
        for b in ev:
            lines.append(f"- {b}")
        cst = (engine or {}).get("compact_stats_text", {})
        last_week = ", ".join(filter(None, [
            cst.get("setup_mix"), cst.get("trigger_quality"), cst.get("calibration_note"), cst.get("risk_discipline")
        ]))
        if last_week:
            lines.append("Last week: " + last_week)
        act = (engine or {}).get("action_hint")
        if act:
            lines.append("What to do: " + act)
        lines.append("")

    # Asset blocks
    drift_warn = float(os.getenv("TB_DIGEST_DRIFT_WARN_PCT", options.get("drift_warn_pct", 0.5)))
    max_tfs = int(os.getenv("TB_DIGEST_MAX_TFS", options.get("max_tfs", 2)))
    include_prices = (os.getenv("TB_DIGEST_INCLUDE_PRICES", "1") == "1") or bool(options.get("include_prices", False))
    include_stock_prices = os.getenv("TB_DIGEST_INCLUDE_STOCK_PRICES", "0") == "1"
    ordered_tfs = ["1h", "4h", "1D", "1W", "1M"]
    TELEGRAM_CRYPTO_ONLY = os.getenv("TB_DIGEST_TELEGRAM_CRYPTO_ONLY", "0") == "1"

    def header(sym: str, a: dict) -> str:
        th = a.get("thesis") or {}
        risk = _hdr_val(th.get("risk_band"), "Medium")
        ready = _hdr_val(th.get("readiness"), "Later")
        act = _hdr_val(th.get("action"), "Watch")
        return f"{sym} — {risk} | {ready} | {act}"

    tf_order = ["1h", "4h", "daily"]

    # Apply crypto-only filter if enabled
    render_symbols = [s for s in assets_ordered if (not TELEGRAM_CRYPTO_ONLY or is_crypto(s))]

    for sym in render_symbols:
        a = assets_data.get(sym, {})
        lines.append(header(sym, a))
        # Evidence line (number-free)
        th = a.get("thesis") or {}
        action = (th.get("action") or a.get("action") or "watch").lower()
        if action in ("buy", "long"): sentiment = "bullish"
        elif action in ("sell", "short"): sentiment = "bearish"
        elif action in ("watch", "neutral"): sentiment = "watch"
        else: sentiment = action or "mixed"
        participation = (a.get("participation") or "normal")
        tf_aligned = bool(a.get("alignment_flag") or th.get("tf_aligned") or False)
        signal_quality = (a.get("signal_quality") or th.get("signal_quality") or "mixed")
        structure_txt = (a.get("structure") or "").lower()
        narrative_tags: List[str] = []
        if "trend" in structure_txt:
            narrative_tags = ["continuation", "trend"]
        elif "range" in structure_txt or "reversion" in structure_txt:
            narrative_tags = ["reversion"]
        ev_line = generate_evidence_line(sentiment, participation, tf_aligned, signal_quality, narrative_tags)
        if ev_line:
            lines.append(ev_line)
        if is_crypto(sym):
            # Spot with drift note
            spot = a.get("spot")
            drift = a.get("drift_since_snapshot_pct")
            if spot is not None:
                if include_prices and isinstance(spot, (int, float)):
                    spot_line = f"Spot: {spot:.2f}"
                    if drift is not None and abs(drift) >= drift_warn:
                        spot_line += f" (re-check; drift >{drift_warn:.1f}%)"
                    lines.append(spot_line)
                else:
                    # Redacted numeric spot
                    if drift is not None and abs(drift) >= drift_warn:
                        lines.append("Spot: price with drift warning")
                    else:
                        lines.append("Spot: price stable")
            # Structure
            if a.get("structure"):
                lines.append("Structure: " + a["structure"])
            # Timeframes
            shown = 0
            for tf in ordered_tfs:
                if shown >= max_tfs:
                    break
                plan = (a.get("plan") or {}).get(tf)
                if not plan:
                    continue
                lines.append(f"{tf}: ")
                # Determine if numeric allowed for this asset (crypto always gated by include_prices; stocks require include_stock_prices)
                allow_numeric = include_prices if is_crypto(sym) else (include_prices and include_stock_prices)
                # Entries can be legacy (number or [lo,hi]) or new schema list of dicts
                ent = plan.get("entries")
                if ent is not None:
                    rendered_entries: List[str] = []
                    if isinstance(ent, list) and ent and isinstance(ent[0], dict):
                        for item in ent:
                            etype = item.get("type")
                            zot = item.get("zone_or_trigger")
                            if allow_numeric and isinstance(zot, (int, float)):
                                s = f"{zot:.2f}"
                            elif allow_numeric and isinstance(zot, (list, tuple)) and len(zot) == 2 and all(isinstance(x,(int,float)) for x in zot):
                                s = f"{zot[0]:.2f}–{zot[1]:.2f}"
                            else:
                                s = "set"
                            if etype:
                                s += f" ({etype})"
                            rendered_entries.append(s)
                    else:
                        # legacy formats
                        if allow_numeric and isinstance(ent, (list, tuple)) and len(ent) == 2 and all(isinstance(x,(int,float)) for x in ent):
                            rendered_entries.append(f"{ent[0]:.2f}–{ent[1]:.2f}")
                        elif allow_numeric and isinstance(ent, (int, float)):
                            rendered_entries.append(f"{ent:.2f}")
                        elif isinstance(ent, str):
                            rendered_entries.append(ent)
                        else:
                            rendered_entries.append("set")
                    if rendered_entries:
                        lines.append("  Entries: " + "; ".join(rendered_entries))
                # Invalidation
                inv = plan.get("invalidation") or plan.get("invalid")
                if inv is not None:
                    if allow_numeric and isinstance(inv, dict) and isinstance(inv.get("price"),(int,float)):
                        cond = inv.get("condition") or "breach"
                        lines.append(f"  Invalidation: {inv['price']:.2f} ({cond})")
                    elif allow_numeric and isinstance(inv, (int, float)):
                        lines.append(f"  Invalidation: {inv:.2f}")
                    else:
                        lines.append("  Invalidation: set")
                # Targets
                tg = plan.get("targets")
                if tg:
                    if allow_numeric:
                        parts = []
                        if isinstance(tg, (list, tuple)):
                            for i, t in enumerate(tg, 1):
                                if isinstance(t, dict) and isinstance(t.get("price"),(int,float)):
                                    label = t.get("label") or f"TP{i}"
                                    parts.append(f"{label} {t['price']:.2f}")
                                elif isinstance(t, (int, float)):
                                    parts.append(f"TP{i} {t:.2f}")
                        if parts:
                            lines.append("  Targets: " + ", ".join(parts))
                    else:
                        lines.append("  Targets: set")
                shown += 1
            # Weekly anchor
            wk = a.get("weekly_anchor") or {}
            if wk.get("supply_zone") or wk.get("demand_zone"):
                parts = []
                if include_prices and isinstance(wk.get("supply_zone"),(list,tuple)) and len(wk["supply_zone"])==2:
                    lo,hi = wk["supply_zone"]
                    parts.append(f"supply {lo:.0f}–{hi:.0f}")
                elif wk.get("supply_zone"):
                    parts.append("supply L–H")
                if include_prices and isinstance(wk.get("demand_zone"),(list,tuple)) and len(wk["demand_zone"])==2:
                    lo,hi = wk["demand_zone"]
                    parts.append(f"demand {lo:.0f}–{hi:.0f}")
                elif wk.get("demand_zone"):
                    parts.append("demand L–H")
                if parts:
                    lines.append("Weekly: " + "; ".join(parts))
            # Sizing and drift guard
            if a.get("sizing_text"):
                lines.append("Sizing: " + a["sizing_text"])
            lines.append("Drift: guard with invalidation")
        else:
            # Stocks: header only plus optional notes
            # Spot (stocks): printed only if stock numeric allowed
            allow_stock_numeric = include_prices and include_stock_prices
            sspot = a.get("spot")
            if allow_stock_numeric and isinstance(sspot, (int, float)):
                lines.append(f"Spot: {sspot:.2f}")
            if a.get("structure"):
                lines.append("Structure: " + a["structure"])
            if a.get("sizing_text"):
                lines.append("Sizing: " + a["sizing_text"])
        lines.append("")

    # Playbook
    playbook = [
        "Trade A-setups only",
        "Let timeframes align",
        "Size with discipline",
    ]
    lines.append("Playbook")
    for p in playbook[:3]:
        lines.append(f"- {p}")

    return "\n".join(lines)
