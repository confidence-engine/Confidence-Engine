from datetime import datetime, timezone
import os
import json
from config import settings
from alpaca import recent_bars, latest_headlines
from coindesk_rss import fetch_coindesk_titles
from pplx_fetcher import fetch_pplx_headlines_with_rotation
from dedupe_utils import dedupe_titles
from narrative_dev import make_from_headlines, filter_relevant_weighted
from finbert import sentiment_robust
from price import price_score
from narrative_analysis import blend, decay
from narrative_analysis_extras import adaptive_trigger
from divergence import compute, reason
from time_utils import minutes_since
from explain import strength_label, volume_label, divergence_label, explain_term
from db import init_db, save_run
from export import export_run_json, save_bars_csv, save_accepted_txt
from alpha_summary import alpha_summary, alpha_next_steps

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def main():
    init_db()
    symbol = settings.symbol
    print(f"[TracerBullet] {symbol} lookback={settings.lookback_minutes}m")

    # 1) Bars
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

    # Source lookup
    alp_set = set(alpaca_heads)
    pplx_set = set(pplx_titles)
    cdx_set = set(coindesk_heads)
    def _origin(h: str) -> str:
        if h in pplx_set: return "perplexity"
        if h in alp_set:  return "alpaca"
        if h in cdx_set:  return "coindesk"
        return "unknown"

    # 4) Relevance gate with per-source weights
    accepted_w, rejected_w = filter_relevant_weighted(
        merged_heads,
        threshold=settings.relevance_threshold,
        source_lookup_fn=_origin,
        weight_overrides=None
    )
    # accepted_w item: (headline, raw_score, weighted_score, source)
    used_heads = [h for (h, _rs, _ws, _src) in accepted_w]
    used_cnt, drop_cnt = len(accepted_w), len(rejected_w)

    # Keyword fallback
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

    # LLM score + confidence
    llm_s = nar.narrative_momentum_score if nar else 0.0
    conf = nar.confidence if nar else (0.50 if used_heads else 0.0)

    # 7) Decay + composite
    last_ts = bars.index[-1].to_pydatetime() if len(bars) else datetime.now(timezone.utc)
    mins_old = minutes_since(last_ts)
    raw_narr = blend(llm_s, fin, 0.6, 0.4)
    dec_narr = decay(raw_narr, mins_old, settings.narrative_halflife_min)

    px, vz = price_score(bars)
    div = compute(dec_narr, px)

    # 8) Adaptive trigger and decision
    base_trig = settings.divergence_threshold
    trig = adaptive_trigger(base_trig, vz)
    default_rc = reason(div, conf, vz, trig, settings.confidence_cutoff)
    rc = "NO_RELEVANT_HEADLINES" if not used_heads else default_rc
    action = "BUY" if (used_heads and abs(div) > trig and conf > settings.confidence_cutoff and vz > -0.5 and div > 0) else "HOLD"

    # 9) Labels
    nar_lbl = strength_label(dec_narr)
    px_lbl = strength_label(px)
    vol_lbl = volume_label(vz)
    gap_lbl = divergence_label(div, trig)

    # 10) Story (audit), alpha-first summary for traders
    if nar:
        story = nar.narrative_summary
    else:
        story = ("; ".join(used_heads)[:200] if used_heads else "No BTC-relevant headlines passed the filter") or "No headlines"

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

    # Alpha-focused outputs
    alpha = alpha_summary(nar_lbl, div, conf, px_lbl, vol_lbl, used_cnt)
    plan = alpha_next_steps(div, conf, trig, vz)

    # Short summary and detail for DB schema compatibility
    summary = (
        f"Narrative {nar_lbl} vs Price {px_lbl}; {gap_lbl}. "
        f"Used {used_cnt}, dropped {drop_cnt}. Action: {action} ({rc})."
    )
    detail = (
        f"FinBERT (relevant-only)={fin:+.2f}, LLM={llm_s:+.2f}, decayed narrative={dec_narr:+.2f}; "
        f"price score={px:+.2f} ({vol_lbl}, z={vz:+.2f}); gap={div:+.2f} "
        f"(trigger>{trig:.2f}); confidence={conf:.2f}."
    )

    # 12) Payload
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
        "summary": summary,
        "detail": detail,

        "alpha_summary": alpha,
        "alpha_next_steps": plan,
        "story_excerpt": story,

        "pplx_provenance": json.dumps(pplx_items, ensure_ascii=False),
        "pplx_last_error": pplx_err or ""
    }

    run_id = save_run(payload)

    # 13) Export artifacts
    try:
        export_run_json(run_id, payload)
        N = 240
        bars_to_save = bars.tail(N) if len(bars) > N else bars
        save_bars_csv(run_id, bars_to_save)
        # Accepted TXT (use weighted relevance for readability)
        lines = [{"source": d["source"], "relevance": d["weighted_relevance"], "headline": d["headline"]} for d in accepted_details]
        save_accepted_txt(run_id, lines)
    except Exception:
        pass

    # 14) Auto-commit
    try:
        paths_to_commit = ["runs", "bars"]
        from autocommit import auto_commit_and_push
        result = auto_commit_and_push(paths_to_commit, extra_message=f"run_id={run_id}", push_enabled=True)
        print(f"\nGit: {result}")
    except Exception:
        pass

    # 15) Console output (alpha-first)
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

    print(f"\nSaved run_id={run_id} to tracer.db")

if __name__ == "__main__":
    main()
