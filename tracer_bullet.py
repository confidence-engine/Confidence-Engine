from datetime import datetime, timezone
import os
import json

# Config and data sources
from config import settings
from alpaca import recent_bars, latest_headlines
from coindesk_rss import fetch_coindesk_titles
from pplx_fetcher import fetch_pplx_headlines_with_rotation
from backtester.ml_gate import predict_prob

# Pipeline utilities
from dedupe_utils import dedupe_titles
from narrative_dev import make_from_headlines, filter_relevant_weighted
from finbert import sentiment_robust
from price import price_score
from narrative_analysis import blend, decay
from narrative_analysis_extras import adaptive_trigger
from divergence import compute, reason
from diversity import compute_diversity_confidence_adjustment
from cascade import detect_cascade
from contrarian import decide_contrarian_viewport
from time_utils import minutes_since
from explain import strength_label, volume_label, divergence_label, explain_term
from timescales import compute_timescale_scores
from confirmation import run_confirmation_checks
from symbol_utils import get_symbol_type
from trading_hours import trading_hours_state

# Persistence and exports
from db import init_db, save_run
from export import export_run_json, save_bars_csv, save_accepted_txt
from retention import prune_artifacts

# Alpha-first summary
from alpha_summary import alpha_summary, alpha_next_steps

# Telegram push (end-of-run)
from telegram_bot import send_message, format_alpha_message
from sizing import map_confidence_to_R

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

"""
Main pipeline orchestrator.

Configuration is sourced from `config.settings`, which in turn reads the
environment (.env and process env). You can override at runtime via either:
- Environment variables: SYMBOL, LOOKBACK_MINUTES, etc.
- CLI wrapper: `python3 scripts/run.py --symbol BTC/USD --lookback 120 --no-telegram`

Telegram sending respects TB_NO_TELEGRAM=1 for safe automation/CI runs.
"""


def main():
    init_db()
    symbol = settings.symbol
    print(f"[TracerBullet] {symbol} lookback={settings.lookback_minutes}m")

    # 1) Market bars
    bars = recent_bars(symbol, settings.lookback_minutes)

    # 2) Headlines
    alpaca_heads = latest_headlines(symbol, settings.headlines_limit)

    pplx_titles, pplx_items, pplx_err = ([], [], None)
    if settings.pplx_enabled and settings.pplx_api_keys:
        pplx_titles, pplx_items, pplx_err = fetch_pplx_headlines_with_rotation(
            settings.pplx_api_keys,
            hours=settings.pplx_hours
        )

    coindesk_heads = fetch_coindesk_titles() if settings.use_coindesk else []

    # 3) Merge + dedupe
    merged_heads = dedupe_titles([*alpaca_heads, *pplx_titles, *coindesk_heads])

    # Source lookup for weighting/provenance
    alp_set = set(alpaca_heads)
    pplx_set = set(pplx_titles)
    cdx_set = set(coindesk_heads)

    def _origin(h: str) -> str:
        if h in pplx_set:
            return "perplexity"
        if h in alp_set:
            return "alpaca"
        if h in cdx_set:
            return "coindesk"
        return "unknown"

    # 4) Relevance gate (per-source weighted)
    accepted_w, rejected_w = filter_relevant_weighted(
        merged_heads,
        threshold=settings.relevance_threshold,
        source_lookup_fn=_origin,
        weight_overrides=None
    )
    # accepted_w item: (headline, raw_score, weighted_score, source)
    used_heads = [h for (h, _rs, _ws, _src) in accepted_w]
    used_cnt, drop_cnt = len(accepted_w), len(rejected_w)

    # Keyword fallback if empty
    if not used_heads:
        kw = ["bitcoin", "btc", "satoshi", "halving"]
        kw_hits = [h for h in merged_heads if any(k in h.lower() for k in kw)]
        if kw_hits:
            seed = kw_hits[:3]
            accepted_w = [(h, 0.0, 0.40, _origin(h)) for h in seed]
            used_heads = seed
            used_cnt, drop_cnt = len(accepted_w), len(rejected_w)

    # 5) Narrative from accepted
    nar = make_from_headlines(used_heads, threshold=settings.relevance_threshold) if used_heads else None

    # 6) Robust sentiment (relevant-only)
    fin, fin_kept, fin_dropped = sentiment_robust(used_heads) if used_heads else (0.0, [], [])

    # 7) Composite narrative and decay
    llm_s = nar.narrative_momentum_score if nar else 0.0
    conf = nar.confidence if nar else (0.50 if used_heads else 0.0)

    last_ts = bars.index[-1].to_pydatetime() if len(bars) else datetime.now(timezone.utc)
    mins_old = minutes_since(last_ts)
    raw_narr = blend(llm_s, fin, 0.6, 0.4)
    dec_narr = decay(raw_narr, mins_old, settings.narrative_halflife_min)

    # 8) Price/volume and divergence
    px, vz = price_score(bars)
    div = compute(dec_narr, px)

    # 9) Trigger, reason, and action
    base_trig = settings.divergence_threshold
    trig = adaptive_trigger(base_trig, vz)
    default_rc = reason(div, conf, vz, trig, settings.confidence_cutoff)
    rc = "NO_RELEVANT_HEADLINES" if not used_heads else default_rc
    action = "BUY" if (used_heads and abs(div) > trig and conf > settings.confidence_cutoff and vz > -0.5 and div > 0) else "HOLD"

    # Optional ML probability gate (env-gated)
    ml_prob = None
    try:
        if os.getenv("TB_USE_ML_GATE", "0").lower() in ("1", "true", "yes", "on"):
            model_path = os.getenv("TB_ML_GATE_MODEL_PATH", "").strip()
            if model_path:
                try:
                    ml_prob = predict_prob(bars, model_path)
                except Exception:
                    ml_prob = None
            min_prob = float(os.getenv("TB_ML_GATE_MIN_PROB", "0.5"))
            # Permissive behavior: if model cannot produce a prob (None), do not block
            if action == "BUY" and (ml_prob is not None and ml_prob < min_prob):
                action = "HOLD"
                rc = "ML_GATE_BLOCKED"
    except Exception:
        pass

    # 10) Labels
    nar_lbl = strength_label(dec_narr)
    px_lbl = strength_label(px)
    vol_lbl = volume_label(vz)
    gap_lbl = divergence_label(div, trig)

    # Story excerpt (for audit)
    story = nar.narrative_summary if nar else (("; ".join(used_heads)[:200] if used_heads else "No BTC-relevant headlines passed the filter") or "No headlines")

    fin_kept_cnt, fin_drop_cnt = len(fin_kept), len(fin_dropped)

    # 11) Relevance details with raw and weighted scores + source
    accepted_details = [
        {"headline": h, "raw_relevance": round(rs, 3), "weighted_relevance": round(ws, 3), "source": src}
        for (h, rs, ws, src) in accepted_w
    ]
    rejected_details = [
        {"headline": h, "raw_relevance": round(rs, 3), "weighted_relevance": round(ws, 3), "source": src}
        for (h, rs, ws, src) in rejected_w
    ]

    # 12) Source diversity adjustment (confidence shaping)
    div_adj, div_meta = compute_diversity_confidence_adjustment(accepted_details)
    conf = min(0.75, max(0.0, conf + div_adj))

    # 13) Cascade detector (hype-only guard)
    cascade = detect_cascade(accepted_details, bars)
    # Apply confidence delta with floor 0.0
    conf = max(0.0, conf + cascade.get("confidence_delta", 0.0))

    # 14) Contrarian viewport (informational)
    # narrative magnitude uses decayed narrative magnitude as proxy
    narr_mag = abs(dec_narr)
    # Use cascade price_move_pct if present; else quick compute from bars
    pm_pct = cascade.get("price_move_pct") if cascade else None
    if pm_pct is None:
        try:
            closes = getattr(bars, "close", None)
            if closes is not None and len(closes) >= 2 and float(closes.iloc[0]) != 0.0:
                pm_pct = abs((float(closes.iloc[-1]) - float(closes.iloc[0])) / float(closes.iloc[0]) * 100.0)
        except Exception:
            pm_pct = 0.0
    pm_pct = float(pm_pct or 0.0)
    contra = decide_contrarian_viewport(narr_mag, div, pm_pct)

    # 15) Multi-timescale scoring (no confidence changes here)
    timescale_scores = compute_timescale_scores(bars, dec_narr)

    # Negative confirmation checks (apply penalty before final capping)
    env_snapshot = dict(os.environ)
    checks, total_penalty = run_confirmation_checks(timescale_scores, env_snapshot)
    conf = max(0.0, conf + total_penalty)

    # 16) Alpha-first outputs
    alpha = alpha_summary(nar_lbl, div, conf, px_lbl, vol_lbl, used_cnt)
    plan = alpha_next_steps(div, conf, trig, vz)

    # 16) Payload (keep summary/detail for DB schema compatibility)
    summary = f"Narrative {nar_lbl} vs Price {px_lbl}; {gap_lbl}. Used {used_cnt}, dropped {drop_cnt}. Action: {action} ({rc})."
    detail = (
        f"FinBERT sentiment (relevant-only)={fin:+.2f} "
        f"[kept={fin_kept_cnt}, dropped_outliers={fin_drop_cnt}], "
        f"LLM={llm_s:+.2f}, decayed narrative={dec_narr:+.2f}; "
        f"price score={px:+.2f} ({vol_lbl}, z={vz:+.2f}); gap={div:+.2f} "
        f"(trigger>{trig:.2f}); confidence={conf:.2f}."
    )

    payload = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "bars_lookback_min": settings.lookback_minutes,
        "headlines_count": len(merged_heads),
        "raw_headlines": json.dumps(merged_heads, ensure_ascii=False),
        "btc_filtered_headlines": json.dumps(used_heads, ensure_ascii=False),
        "relevance_details": json.dumps({"accepted": accepted_details, "rejected": rejected_details}, ensure_ascii=False),

        "finbert_score": float(fin),
        "finbert_kept_count": fin_kept_cnt,
        "finbert_dropped_count": fin_drop_cnt,
        "finbert_kept_scores": json.dumps(fin_kept),
        "finbert_dropped_scores": json.dumps(fin_dropped),

        "llm_score": float(llm_s),
        "raw_narrative": float(raw_narr),
        "decayed_narrative": float(dec_narr),
        "price_score": float(px),
        "volume_z": float(vz),
        "divergence": float(div),
        "divergence_threshold": float(trig),
        "confidence": float(conf),
        "action": action,
        "reason": rc,

        "alpha_summary": alpha,
        "alpha_next_steps": plan,
        "story_excerpt": story,
        "source_diversity": {**div_meta, "adjustment": round(div_adj, 3)},
        "cascade_detector": cascade,
        "contrarian_viewport": contra,
        "timescale_scores": timescale_scores,
        "confirmation_checks": checks,
        "confirmation_penalty": float(round(total_penalty, 3)),

        # ML gate telemetry (if enabled)
        "ml_prob": (float(ml_prob) if ml_prob is not None else None),

        "summary": summary,
        "detail": detail,

        "pplx_provenance": json.dumps(pplx_items, ensure_ascii=False),
        "pplx_last_error": pplx_err or ""
    }

    # Add symbol type and market hours state
    symbol_type = get_symbol_type(symbol)
    market_hours = trading_hours_state(symbol)
    payload["symbol_type"] = symbol_type
    payload["market_hours_state"] = market_hours["state"]

    # Position sizing (informational only in V3)
    try:
        payload["position_sizing"] = map_confidence_to_R(payload["confidence"], vol_norm=None)
    except Exception:
        payload["position_sizing"] = {"confidence": payload["confidence"], "target_R": 0.0, "notes": ["error"], "params": {}}

    run_id = save_run(payload)

    # 17) Export artifacts
    try:
        export_run_json(run_id, payload)
        N = 240
        bars_to_save = bars.tail(N) if len(bars) > N else bars
        save_bars_csv(run_id, bars_to_save)
        lines = [{"source": d["source"], "relevance": d["weighted_relevance"], "headline": d["headline"]} for d in accepted_details]
        save_accepted_txt(run_id, lines)
    except Exception:
        pass

    # 18) Auto-commit
    try:
        paths_to_commit = ["runs", "bars"]
        from autocommit import auto_commit_and_push
        result = auto_commit_and_push(paths_to_commit, extra_message=f"run_id={run_id}", push_enabled=True)
        print(f"\nGit: {result}")
    except Exception:
        pass

    # 19) Console output (alpha-first)
    print(f"Bars={len(bars)} Headlines={len(merged_heads)} (used {used_cnt}, dropped {drop_cnt})")
    print("\n[Accepted (source, raw→weighted)]")
    for d in accepted_details[:10]:
        print(f"- {d['source']} | {d['raw_relevance']:.3f}→{d['weighted_relevance']:.3f} | {d['headline'][:120]}")

    all_scored = accepted_w + rejected_w
    top5 = sorted(all_scored, key=lambda x: x[2], reverse=True)[:5]
    print("\n[Relevance top-5 (weighted)]")
    for h, rs, ws, src in top5:
        print(f"- {src} | {rs:.3f}→{ws:.3f} | {h[:120]}")

    print("\n=== Alpha Summary ===")
    print(alpha)
    print("\n" + plan)

    # 18) Retention hygiene (best-effort)
    try:
        keep = int(os.getenv("TB_ARTIFACTS_KEEP", "500"))
        prune_artifacts(keep=keep)
    except Exception:
        pass

    # 19) Telegram push (end-of-run)
    try:
        if os.getenv("TB_NO_TELEGRAM", "0").lower() in ("1", "true", "yes", "on"):
            print("\n[Telegram] skipped (TB_NO_TELEGRAM=1)")
        else:
            msg = format_alpha_message(payload)
            ok = send_message(msg)
            print(f"\n[Telegram] sent: {ok}")
    except Exception as e:
        print("[Telegram] error:", e)

    print(f"\nSaved run_id={run_id} to tracer.db")


if __name__ == "__main__":
    main()
