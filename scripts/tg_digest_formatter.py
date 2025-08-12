#!/usr/bin/env python3
"""
Telegram human digest formatter (plain text, narrative-first).
Adds Weekly Overview and Engine in One Minute sections and per-asset blocks.
"""
from typing import Dict, List
import os

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

    def header(sym: str, a: dict) -> str:
        th = a.get("thesis") or {}
        risk = _hdr_val(th.get("risk_band"), "Medium")
        ready = _hdr_val(th.get("readiness"), "Later")
        act = _hdr_val(th.get("action"), "Watch")
        return f"{sym} — {risk} | {ready} | {act}"

    tf_order = ["1h", "4h", "daily"]

    for sym in assets_ordered:
        a = assets_data.get(sym, {})
        lines.append(header(sym, a))
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
            for tf in tf_order:
                if shown >= max_tfs:
                    break
                plan = (a.get("plan") or {}).get(tf)
                if not plan:
                    continue
                lines.append(f"{tf}: ")
                # Entries: support zones (L–H) or single triggers
                ent = plan.get("entries")
                if ent is not None:
                    if include_prices and isinstance(ent, (list, tuple)) and len(ent) == 2 and all(isinstance(x,(int,float)) for x in ent):
                        lines.append(f"  Entries: {ent[0]:.2f}–{ent[1]:.2f}")
                    elif include_prices and isinstance(ent, (int, float)):
                        lines.append(f"  Entries: {ent:.2f}")
                    else:
                        # narrative fallback
                        lines.append("  Entries: set")
                # Invalidation
                inv = plan.get("invalidation") or plan.get("invalid")
                if inv is not None:
                    if include_prices and isinstance(inv, dict) and isinstance(inv.get("price"),(int,float)):
                        cond = inv.get("condition") or "breach"
                        lines.append(f"  Invalidation: {inv['price']:.2f} ({cond})")
                    elif include_prices and isinstance(inv, (int, float)):
                        lines.append(f"  Invalidation: {inv:.2f}")
                    else:
                        lines.append("  Invalidation: set")
                # Targets
                tg = plan.get("targets")
                if tg:
                    if include_prices:
                        parts = []
                        if isinstance(tg, (list, tuple)):
                            for i, t in enumerate(tg, 1):
                                if isinstance(t, dict) and isinstance(t.get("price"),(int,float)):
                                    parts.append(f"TP{i} {t['price']:.2f}")
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
