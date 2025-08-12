#!/usr/bin/env python3
"""
Weekly and Engine narrative builders for the Telegram human digest.
Minimal heuristics over existing payloads; no numbers/probabilities.
"""
from typing import Dict, List

# --- Weekly Overview ---

def _infer_regime(assets_data: Dict[str, dict]) -> str:
    # Use cross-timeframe alignment if available
    aligned = []
    combined = []
    for a in assets_data.values():
        ts = a.get("timescale_scores", {})
        aligned.append(ts.get("aligned_horizons"))
        combined.append(ts.get("combined_divergence"))
    score = sum(1 for x in aligned if (x or 0) >= 2)
    bias = sum(1 for c in combined if (c or 0) > 0) - sum(1 for c in combined if (c or 0) < 0)
    if score >= max(1, len(assets_data)//2) and bias > 0:
        return "risk-on"
    if score >= max(1, len(assets_data)//2) and bias < 0:
        return "risk-off"
    return "mixed"

def build_weekly_overview(assets_data: Dict[str, dict]) -> dict:
    anchors: List[dict] = []
    # Include up to two anchors if any asset provides weekly zones
    for sym, a in assets_data.items():
        wk = a.get("weekly_anchor") or {}
        if wk.get("supply_zone") or wk.get("demand_zone"):
            note = wk.get("note") or "key weekly zone"
            if wk.get("supply_zone"):
                anchors.append({"type": "supply", "zone": wk["supply_zone"], "note": note})
            if len(anchors) >= 2:
                break
            if wk.get("demand_zone") and len(anchors) < 2:
                anchors.append({"type": "demand", "zone": wk["demand_zone"], "note": note})
        if len(anchors) >= 2:
            break

    regime = _infer_regime(assets_data)
    plan_text = "Stay selective; lean with the prevailing flow, fade extremes, respect invalidation." if regime != "mixed" else "Focus on clarity; wait for alignment and defined risk."

    return {
        "regime": regime,
        "anchors": anchors[:2],
        "plan_text": plan_text,
        "catalysts": []
    }

# --- Engine in One Minute ---

def build_engine_minute(assets_data: Dict[str, dict]) -> dict:
    # High-level thesis from BTC/ETH bias if present
    bias_counts = {"up": 0, "down": 0}
    for a in assets_data.values():
        b = (a.get("thesis") or {}).get("bias")
        if b == "up":
            bias_counts["up"] += 1
        elif b == "down":
            bias_counts["down"] += 1
    if bias_counts["up"] > bias_counts["down"]:
        thesis = "Bulls hold the initiative; lean into clean triggers, avoid chasing."
    elif bias_counts["down"] > bias_counts["up"]:
        thesis = "Supply has the edge; sell rallies into structure, keep risk tight."
    else:
        thesis = "Tape is balanced; wait for leaders to speak, trade the break or the fail."

    evidence_bullets = []
    # Up to 2 short bullets from alignment and structure
    any_asset = next(iter(assets_data.values()), {})
    ts = any_asset.get("timescale_scores", {})
    if ts.get("aligned_horizons", 0) >= 2:
        evidence_bullets.append("Timeframes are aligning on leaders")
    if any_asset.get("structure"):
        evidence_bullets.append("Structure is defined; plan is repeatable")
    evidence_bullets = evidence_bullets[:2]

    compact_stats_text = {
        "setup_mix": "blend of breakouts and mean reversion",
        "trigger_quality": "clean when alignment holds",
        "calibration_note": "prefer A-setups; pass on noise",
        "risk_discipline": "define invalidation upfront"
    }

    action_hint = "Trade the plan, not the hope; scale with confirmation."

    return {
        "thesis_text": thesis,
        "evidence_bullets": evidence_bullets,
        "compact_stats_text": compact_stats_text,
        "action_hint": action_hint,
    }
