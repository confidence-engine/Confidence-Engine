import os
from typing import List, Dict, Any
from .discord_sender import MAX_DESC_CHARS
from .evidence_lines import generate_evidence_line


def _hdr_val(v, default):
    s = str(v or "").strip()
    return s if s else default


def digest_to_discord_embeds(digest_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map structured digest_data to Discord embeds.

    Expected digest_data shape:
    {
        "timestamp": str,
        "executive_take": str | None,
        "weekly": str,
        "engine": str,
        "assets": [
            {
                "symbol": str,
                "spot": float | None,
                "structure": str | None,
                "risk": str,
                "readiness": str,
                "action": str,
                "plan": { tf: { entries: [...], invalidation: {...}, targets: [...] } }
            }, ...
        ]
    }
    """
    embeds: List[Dict[str, Any]] = []

    # Header / Executive Take
    exec_raw = digest_data.get("executive_take")
    if isinstance(exec_raw, str):
        header_desc = exec_raw.strip()
    else:
        header_desc = str(exec_raw) if exec_raw is not None else ""
    embeds.append({
        "title": f"Tracer Bullet — {digest_data.get('timestamp','')}",
        "description": header_desc[:MAX_DESC_CHARS],
        "color": 0xFFD700,
    })

    # Weekly + Engine as fields
    def _format_weekly(w):
        if isinstance(w, dict):
            lines = []
            regime = w.get("regime")
            if regime:
                lines.append(f"Regime: {regime}")
            plan_text = w.get("plan_text")
            if plan_text:
                lines.append(f"Weekly Plan: {plan_text}")
            anchors = w.get("anchors") or []
            if anchors:
                lines.append("Anchors:")
                for a in anchors:
                    lines.append(f"- {a}")
            catalysts = w.get("catalysts") or []
            if catalysts:
                lines.append("Catalysts:")
                for c in catalysts:
                    lines.append(f"- {c}")
            return "\n".join(lines)
        # fallback
        return w.strip() if isinstance(w, str) else (str(w) if w is not None else "")

    def _format_engine(e):
        if isinstance(e, dict):
            lines = []
            th = e.get("thesis_text")
            if th:
                lines.append(f"Thesis: {th}")
            bullets = e.get("evidence_bullets") or []
            for b in bullets:
                lines.append(f"- {b}")
            stats = e.get("compact_stats_text") or {}
            if isinstance(stats, dict) and stats:
                # Show a compact one-line summary
                mix = stats.get("setup_mix")
                trig = stats.get("trigger_quality")
                cal = stats.get("calibration_note")
                risk = stats.get("risk_discipline")
                parts = [p for p in [mix, trig, cal, risk] if p]
                if parts:
                    lines.append("; ".join(parts))
            hint = e.get("action_hint")
            if hint:
                lines.append(f"What to do: {hint}")
            return "\n".join(lines)
        # fallback
        return e.strip() if isinstance(e, str) else (str(e) if e is not None else "")

    weekly = _format_weekly(digest_data.get("weekly"))
    engine = _format_engine(digest_data.get("engine"))
    embeds.append({
        "title": "Weekly Overview & Engine",
        "fields": [
            {"name": "Weekly", "value": weekly or "-", "inline": False},
            {"name": "Engine", "value": engine or "-", "inline": False},
        ],
        "color": 0x00BFFF,
    })

    # Polymarket BTC/ETH (optional)
    pm_list = digest_data.get("polymarket") or []
    show_empty = os.getenv("TB_POLYMARKET_SHOW_EMPTY", "0") == "1"
    if pm_list or show_empty:
        pm_fields: List[Dict[str, Any]] = []
        max_items = int(os.getenv("TB_POLYMARKET_MAX_ITEMS", "2"))
        if pm_list:
            for pm in pm_list[:max_items]:
                title = pm.get("title") or "Crypto market"
                stance = pm.get("stance") or "Stand Aside"
                readiness = pm.get("readiness") or "Later"
                edge = pm.get("edge_label") or "in-line"
                val_lines = [f"{stance} | {readiness} | {edge}"]
                rat = pm.get("rationale_chat")
                if rat:
                    val_lines.append(rat)
                if os.getenv("TB_POLYMARKET_SHOW_OUTCOME", "1") == "1":
                    out_label = pm.get("outcome_label") or pm.get("implied_side") or "-"
                    if os.getenv("TB_POLYMARKET_SHOW_PROB", "0") == "1":
                        pct = pm.get("implied_pct")
                        try:
                            if isinstance(pct, int):
                                out_label = f"{out_label} ({pct}%)"
                            else:
                                imp = pm.get("implied_prob")
                                if isinstance(imp,(int,float)):
                                    out_label = f"{out_label} ({float(imp)*100:.0f}%)"
                        except Exception:
                            pass
                        try:
                            ip = pm.get("internal_prob")
                            imp = pm.get("implied_prob")
                            if isinstance(ip,(int,float)) and isinstance(imp,(int,float)):
                                ipct = round(float(ip)*100)
                                mpct = round(float(imp)*100)
                                if abs(ipct - mpct) >= 3:
                                    out_label += f" | Model: {ipct}%"
                        except Exception:
                            pass
                    val_lines.append(f"Outcome: {out_label}")
                pm_fields.append({
                    "name": title,
                    "value": "\n".join(val_lines) or "-",
                    "inline": False,
                })
        else:
            pm_fields.append({"name": "-", "value": "No qualifying BTC/ETH markets today.", "inline": False})
        embeds.append({
            "title": "Polymarket BTC/ETH",
            "fields": pm_fields or [{"name": "-", "value": "-", "inline": False}],
            "color": 0x8A2BE2,
        })

    # Assets
    for asset in digest_data.get("assets", []):
        fields = []
        # Evidence line (number-free)
        th = asset.get("thesis") or {}
        action = (th.get("action") or asset.get("action") or "watch").lower()
        if action in ("buy", "long"):
            sentiment = "bullish"
        elif action in ("sell", "short"):
            sentiment = "bearish"
        elif action in ("watch", "neutral"):
            sentiment = "watch"
        else:
            sentiment = action or "mixed"
        participation = (asset.get("participation") or "normal")
        tf_aligned = bool(asset.get("alignment_flag") or th.get("tf_aligned") or False)
        signal_quality = (asset.get("signal_quality") or th.get("signal_quality") or "mixed")
        structure_txt = (asset.get("structure") or "").lower()
        narrative_tags: List[str] = []
        if "trend" in structure_txt:
            narrative_tags = ["continuation", "trend"]
        elif "range" in structure_txt or "reversion" in structure_txt:
            narrative_tags = ["reversion"]
        ev_line = generate_evidence_line(sentiment, participation, tf_aligned, signal_quality, narrative_tags)
        if ev_line:
            fields.append({"name": "Evidence", "value": ev_line, "inline": False})
        plan = asset.get("plan") or {}
        # Preserve insertion order if dicts are ordered; otherwise display sorted by common TF order
        tf_order = ["1h", "4h", "1D", "1W", "1M"]
        keys = [k for k in tf_order if k in plan] + [k for k in plan.keys() if k not in tf_order]
        for tf in keys:
            p = plan.get(tf) or {}
            field_val_parts = []
            entries = p.get("entries") or []
            invalid = p.get("invalidation") or {}
            targets = p.get("targets") or []
            if entries:
                def _fmt_entry(it):
                    t = it.get("type") or "entry"
                    z = it.get("zone_or_trigger")
                    if isinstance(z, (list, tuple)) and len(z) == 2:
                        return f"{z[0]:.2f}–{z[1]:.2f} ({t})"
                    if isinstance(z, (int, float)):
                        return f"{z:.2f} ({t})"
                    return str(z)
                field_val_parts.append("Entries: " + ", ".join(_fmt_entry(x) for x in entries))
            if invalid:
                inv_price = invalid.get("price")
                inv_cond = invalid.get("condition")
                if inv_price is not None:
                    field_val_parts.append(f"Invalid: {inv_price:.2f} ({inv_cond or ''})".strip())
            if targets:
                tgt_str = ", ".join([f"{t.get('label','TP')} {t.get('price',0):.2f}" for t in targets])
                field_val_parts.append("Targets: " + tgt_str)
            fields.append({"name": str(tf), "value": ("\n".join(field_val_parts) or "-"), "inline": False})
        title = f"{asset.get('symbol','')} — " \
                f"{_hdr_val(asset.get('risk'),'Medium')} | " \
                f"{_hdr_val(asset.get('readiness'),'Later')} | " \
                f"{_hdr_val(asset.get('action'),'Watch')}"
        desc = []
        if asset.get("spot") is not None:
            try:
                desc.append(f"Spot: {float(asset['spot']):.2f}")
            except Exception:
                desc.append(f"Spot: {asset['spot']}")
        if asset.get("structure"):
            desc.append(f"Structure: {asset['structure']}")
        embeds.append({
            "title": title,
            "description": "\n".join(desc)[:MAX_DESC_CHARS],
            "fields": fields,
            "color": 0x32CD32,
        })

    return embeds
