import os
from typing import List, Dict, Any
from .discord_sender import MAX_DESC_CHARS
from .evidence_lines import generate_evidence_line, generate_high_risk_note, estimate_confidence_pct


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

    # Helper: basic crypto detector by symbol naming
    CRYPTO_PREFIXES = (
        "BTC/", "ETH/", "SOL/", "ADA/", "BNB/", "XRP/", "DOGE/", "DOT/", "LTC/", "AVAX/",
    )
    def _is_crypto(sym: str) -> bool:
        if not sym:
            return False
        s = str(sym).upper()
        if any(stable in s for stable in ("/USD", "/USDT", "/USDC")):
            return True
        return s.startswith(CRYPTO_PREFIXES)

    # Helpers for simple English and A+ heuristic
    def _simple_english(text):
        if not isinstance(text, str):
            return text
        repl = {
            "lean into clean triggers": "take clear, simple trades",
            "avoid chasing": "don't buy late",
            "scale with confirmation": "add only after it proves itself",
            "trade the break or the fail": "trade the clear move or the clear rejection",
            "defined risk": "small, known risk",
            "invalidation": "exit plan",
        }
        out = text
        for k, v in repl.items():
            out = out.replace(k, v)
        return out

    def _is_aplus_setup(asset: Dict[str, Any]) -> bool:
        if not isinstance(asset, dict):
            return False
        th = asset.get("thesis") or {}
        action = (th.get("action") or asset.get("action") or "").lower()
        if action not in ("buy", "long", "sell", "short"):
            return False
        readiness = (th.get("readiness") or asset.get("readiness") or "").lower()
        if readiness not in ("now", "near"):
            return False
        # Alignment: accept any of the known flags, including nested timescale_scores.alignment_flag
        tss = asset.get("timescale_scores") or {}
        aligned = bool(
            asset.get("alignment_flag")
            or th.get("tf_aligned")
            or (isinstance(tss, dict) and tss.get("alignment_flag"))
        )
        if not aligned:
            return False
        # Signal quality: prefer explicit field; otherwise infer from confirmation_checks
        sigq = (asset.get("signal_quality") or th.get("signal_quality") or "").lower()
        if not sigq:
            checks = asset.get("confirmation_checks") or []
            try:
                passed = {str(c.get("name")): bool(c.get("passed")) for c in checks if isinstance(c, dict)}
                pv = passed.get("price_vs_narrative", False)
                vol = passed.get("volume_support", False)
                tfa = passed.get("timescale_alignment", False)
                if pv and vol and (tfa or aligned):
                    sigq = "strong"
                elif pv or vol:
                    sigq = "elevated"
                else:
                    sigq = "mixed"
            except Exception:
                sigq = "mixed"
        if sigq not in ("strong", "elevated"):
            return False
        risk_band = (th.get("risk_band") or asset.get("risk") or asset.get("risk_band") or "").lower()
        if risk_band == "high":
            return False
        return True

    # Header / Executive Take
    exec_raw = digest_data.get("executive_take")
    if isinstance(exec_raw, str):
        header_desc = _simple_english(exec_raw.strip())
    else:
        header_desc = str(exec_raw) if exec_raw is not None else ""
    # Executive vs leaders alignment note
    try:
        weekly = digest_data.get("weekly") or {}
        regime = (weekly.get("regime") if isinstance(weekly, dict) else "")
        regime_l = (regime or "").strip().lower()
        leaders = []
        assets = digest_data.get("assets") or []
        for a0 in assets[:2]:
            th0 = (a0.get("thesis") or {}) if isinstance(a0, dict) else {}
            act0 = (th0.get("action") or a0.get("action") or "").strip().lower()
            if act0:
                leaders.append(act0)
        if regime_l in {"mixed", "balanced"} and len(leaders) >= 2:
            if all(x in ("buy", "long") for x in leaders[:2]):
                header_desc = (header_desc + ("\n" if header_desc else "") + "Leaders skew long; wait for clean triggers.").strip()
            elif any(x in ("buy", "long") for x in leaders[:2]) and any(x in ("sell", "short") for x in leaders[:2]):
                header_desc = (header_desc + ("\n" if header_desc else "") + "Leaders diverge from tape; trade only A-setups.").strip()
    except Exception:
        pass
    # Provenance (artifact + git short SHA)
    prov = (digest_data.get("provenance") or {}) if isinstance(digest_data, dict) else {}
    prov_str = ""
    try:
        art = prov.get("artifact")
        sha = prov.get("git")
        if art or sha:
            prov_str = "\n" + ("Source: " + " ".join([x for x in [art, ("@ " + sha) if sha else None] if x]))
    except Exception:
        pass
    embeds.append({
        "title": f"Tracer Bullet — {digest_data.get('timestamp','')}",
        "description": (header_desc + prov_str)[:MAX_DESC_CHARS],
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
                lines.append(f"Weekly Plan: {_simple_english(plan_text)}")
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
                lines.append(f"Thesis: {_simple_english(th)}")
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
                # Optional internal confidence (agent view)
                if os.getenv("TB_POLYMARKET_SHOW_CONFIDENCE", "0") == "1":
                    ip = pm.get("internal_prob")
                    try:
                        if isinstance(ip,(int,float)):
                            val_lines.append(f"Confidence: {round(float(ip)*100)}% (internal)")
                    except Exception:
                        pass
                if os.getenv("TB_POLYMARKET_SHOW_OUTCOME", "1") == "1" and os.getenv("TB_POLYMARKET_NUMBERS_IN_CHAT", "0") == "1":
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
        # Optional: hide equities if no live provider (spot None)
        try:
            if os.getenv("TB_DIGEST_HIDE_EQUITIES_IF_NO_DATA", "1") == "1":
                sym = asset.get("symbol") or ""
                if (not _is_crypto(sym)) and (asset.get("spot") is None):
                    continue
        except Exception:
            pass
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
        # Optional graded risk note for High Risk + Buy/Watch
        note = ""
        if int(os.getenv("TB_DIGEST_EXPLAIN_HIGH_RISK_ACTION", 1)):
            risk_band = asset.get("risk") or asset.get("risk_band") or (th.get("risk_band") if isinstance(th, dict) else None)
            action_lbl = asset.get("action") or (th.get("action") if isinstance(th, dict) else None)
            risk_score = asset.get("risk_score") or (th.get("risk_score") if isinstance(th, dict) else 0)
            note = generate_high_risk_note(risk_band, action_lbl, risk_score)
        if ev_line or note:
            val = ev_line or ""
            if note:
                val = (val + ("\n" if val else "") + f"⚠ {note}").strip()
            fields.append({"name": "Evidence", "value": val, "inline": False})
        # Optional agent confidence (assets)
        if os.getenv("TB_SHOW_ASSET_CONFIDENCE_IN_CHAT", "0") == "1":
            try:
                risk_band = (th.get("risk_band") if isinstance(th, dict) else None) or asset.get("risk") or asset.get("risk_band")
                conf = estimate_confidence_pct(signal_quality, tf_aligned, risk_band)
                fields.append({"name": "Confidence", "value": f"{round(conf*100)}% (agent)", "inline": True})
            except Exception:
                pass
        plan = asset.get("plan") or {}
        # Preserve insertion order if dicts are ordered; otherwise display sorted by common TF order
        tf_order = ["1h", "4h", "1D", "1W"]
        keys = [k for k in tf_order if k in plan] + [k for k in plan.keys() if k not in tf_order]
        for tf in keys:
            p = plan.get(tf) or {}
            field_val_parts = []
            entries = p.get("entries") or []
            invalid = p.get("invalidation") or {}
            targets = p.get("targets") or []
            # provenance hint
            src = p.get("source")
            tf_label = str(tf)
            if src == "analysis":
                tf_label = f"{tf_label} (agent mode)"
            elif src == "fallback":
                tf_label = f"{tf_label} (fallback)"
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
            # Why (number-free analysis explanation)
            try:
                why = p.get("explain")
                if isinstance(why, str) and why.strip():
                    field_val_parts.append("Why: " + why.strip())
            except Exception:
                pass
            fields.append({"name": tf_label, "value": ("\n".join(field_val_parts) or "-"), "inline": False})
        title = f"{asset.get('symbol','')} — " \
                f"Risk Level: {_hdr_val(asset.get('risk'),'Medium')} | " \
                f"Timing: {_hdr_val(asset.get('readiness'),'Later')} | " \
                f"Stance: {_hdr_val(asset.get('action'),'Watch')}"
        if _is_aplus_setup(asset):
            title += "  [A+ Setup]"
        desc = []
        if asset.get("spot") is not None:
            try:
                desc.append(f"Spot: {float(asset['spot']):.2f}")
            except Exception:
                desc.append(f"Spot: {asset['spot']}")
        if asset.get("structure"):
            desc.append(f"Pattern: {asset['structure']}")
        embeds.append({
            "title": title,
            "description": "\n".join(desc)[:MAX_DESC_CHARS],
            "fields": fields,
            "color": 0x32CD32,
        })

    # Quick Summary (simple English) as a final embed
    try:
        qs = _render_quick_summary(
            digest_data.get("weekly") or {},
            digest_data.get("engine") or {},
            digest_data.get("assets") or [],
            digest_data.get("polymarket") or [],
        )
        if qs:
            embeds.append({
                "title": "Quick Summary",
                "description": qs[:MAX_DESC_CHARS],
                "color": 0xFFA500,
            })
    except Exception:
        pass

    return embeds


def _render_quick_summary(weekly: Any, engine: Any, assets: List[Dict[str, Any]], polymarket: List[Dict[str, Any]]) -> str:
    """
    Build a short, kid-friendly multi-line summary in plain English for Discord.
    Avoid numbers; focus on simple takeaways from engine, weekly, polymarket, and top assets.
    """
    def _simple_action(a: str) -> str:
        a = (a or "").lower()
        if a in ("buy", "long"): return "going up"
        if a in ("sell", "short"): return "going down"
        if a in ("watch", "neutral"): return "sideways"
        return "mixed"

    def _name(sym: str) -> str:
        if not sym:
            return ""
        s = str(sym).upper()
        if s.startswith("BTC/"): return "Bitcoin"
        if s.startswith("ETH/"): return "Ethereum"
        return s.split("/")[0]

    parts: List[str] = []
    th = engine.get("thesis_text") if isinstance(engine, dict) else None
    rg = weekly.get("regime") if isinstance(weekly, dict) else None
    if isinstance(th, str) and th.strip():
        parts.append(f"Big picture: {th.strip()}")
    elif isinstance(rg, str) and rg.strip():
        parts.append(f"Big picture: market looks {rg.strip().lower()} today.")
    parts.append("Plan: Trade only clean setups. Keep risk small when things are messy. Always know your exit.")

    # Leaders (top 2)
    leader_lines: List[str] = []
    def _leader_note(act: str) -> str:
        act = (act or "").lower()
        if act in ("buy", "long"): return "look for momentum and alignment before buying"
        if act in ("sell", "short"): return "sell pops into tough areas if they fail"
        return "be patient and wait for a clean move"
    for a in (assets or [])[:2]:
        try:
            sym = a.get("symbol")
            th0 = a.get("thesis") or {}
            act = th0.get("action") or a.get("action")
            label = _name(sym)
            if label and act:
                leader_lines.append(f"- {label}: looks {_simple_action(act)} — {_leader_note(act)}.")
        except Exception:
            continue
    if leader_lines:
        parts.append("Leaders:")
        parts.extend(leader_lines)

    # Polymarket summary
    try:
        pm = polymarket or []
        view = ""
        if pm:
            has_rich = any("rich" in str(x.get("edge_label", "")) for x in pm)
            has_cheap = any("cheap" in str(x.get("edge_label", "")) for x in pm)
            near = any((x.get("readiness") or "").lower() == "near" for x in pm)
            if has_rich:
                view = "crowd paying rich for up bets; looks stretched. Wait for clean triggers."
            elif has_cheap:
                view = "crowd underpricing some moves; look for solid triggers."
            elif near:
                view = "timing is getting close; watch for clear triggers."
        if view:
            parts.append("Polymarket:")
            parts.append(f"- {view}")
    except Exception:
        pass

    # Coins today (top few)
    coin_lines: List[str] = []
    def _readiness_note(r: str) -> str:
        r = (r or "").lower()
        if r == "now": return "ready now"
        if r == "near": return "getting close"
        return "later / be patient"
    for a in (assets or [])[:6]:
        try:
            sym = a.get("symbol")
            th0 = a.get("thesis") or {}
            act = th0.get("action") or a.get("action")
            ready = th0.get("readiness") or a.get("readiness")
            if not act:
                continue
            label = _name(sym)
            tag = " (A+)" if _is_aplus_setup(a) else ""
            coin_lines.append(f"- {label}: {_simple_action(act)} — {_readiness_note(ready)}.{tag}")
        except Exception:
            continue
    if coin_lines:
        parts.append("Coins today:")
        parts.extend(coin_lines)

    # How to trade
    parts.append("How to trade this:")
    parts.append("- Take only A+ setups that match both story and price.")
    parts.append("- If the move breaks, exit quickly; if it confirms, add slowly.")
    parts.append("- Use the invalidation as your safety line so losses stay small.")

    return "\n".join([p for p in parts if isinstance(p, str) and p.strip()])
