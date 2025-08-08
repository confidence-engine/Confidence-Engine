from datetime import datetime, timezone
import os
import json
from config import settings
from alpaca import recent_bars, latest_headlines
from coindesk_rss import fetch_coindesk_titles
from pplx_fetcher import fetch_pplx_headlines_with_rotation
from dedupe_utils import dedupe_titles
from narrative_dev import make_from_headlines, filter_relevant
from finbert import sentiment_robust
from price import price_score
from narrative_analysis import blend, decay
from narrative_analysis_extras import adaptive_trigger
from divergence import compute, reason
from time_utils import minutes_since
from explain import strength_label, volume_label, divergence_label, explain_term
from db import init_db, save_run
from export import export_run_json, save_bars_csv, save_accepted_txt
from autocommit import auto_commit_and_push
from provenance import tag_origins

# Silence tokenizer fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def main():
    init_db()
    symbol = settings.symbol
    print(f"[TracerBullet] {symbol} lookback={settings.lookback_minutes}m")

    # 1) Bars
    bars = recent_bars(symbol, settings.lookback_minutes)

    # 2) Headlines: Alpaca + Perplexity (+ CoinDesk if enabled)
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

    # 4) Relevance gate
    accepted, rejected = filter_relevant(merged_heads, threshold=settings.relevance_threshold)
    used_heads = [h for (h, _s) in accepted]
    used_cnt, drop_cnt = len(accepted), len(rejected)

    # Optional fallback if none accepted
    if not used_heads:
        kw = ["bitcoin", "btc", "satoshi", "halving"]
        kw_hits = [h for h in merged_heads if any(k in h.lower() for k in kw)]
        if kw_hits:
            seed = kw_hits[:3]
            accepted = [(h, 0.40) for h in seed]
            used_heads = seed
            used_cnt, drop_cnt = len(accepted), len(rejected)

    # 5) Narrative from accepted
    nar = make_from_headlines([h for (h, _s) in accepted], threshold=settings.relevance_threshold) if accepted else None

    # 6) Robust sentiment (relevant-only)
    fin, fin_kept, fin_dropped = sentiment_robust(used_heads) if used_heads else (0.0, [], [])

    # Confidence fallback if nar missing but accepted exists
    llm_s = nar.narrative_momentum_score if nar else 0.0
    conf = nar.confidence if nar else (0.50 if accepted else 0.0)

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

    # 10) Story
    if nar:
        story = nar.narrative_summary
    else:
        story = ("; ".join([h for (h, _s) in accepted])[:200] if accepted else "No BTC-relevant headlines passed the filter") or "No headlines"

    fin_kept_cnt, fin_drop_cnt = len(fin_kept), len(fin_dropped)

    # 11) Provenance tagging for accepted items
    accepted_with_src = tag_origins(accepted, alpaca_heads, pplx_titles, coindesk_heads)

    summary = f"Narrative {nar_lbl} vs Price {px_lbl}; {gap_lbl}. Used {used_cnt}, dropped {drop_cnt}. Action: {action} ({rc})."
    detail = (
        f"FinBERT sentiment (relevant-only)={fin:+.2f} "
        f"[kept={fin_kept_cnt}, dropped_outliers={fin_drop_cnt}], "
        f"LLM={llm_s:+.2f}, decayed narrative={dec_narr:+.2f}; "
        f"price score={px:+.2f} ({vol_lbl}, z={vz:+.2f}); gap={div:+.2f} "
        f"(trigger>{trig:.2f}); confidence={conf:.2f}."
    )

    # 12) Payload
    relevance_details = {
        "accepted": accepted_with_src,
        "rejected": [{"headline": h, "relevance": round(s, 3)} for (h, s) in rejected]
    }

    payload = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "bars_lookback_min": settings.lookback_minutes,
        "headlines_count": len(merged_heads),
        "raw_headlines": json.dumps(merged_heads, ensure_ascii=False),
        "btc_filtered_headlines": json.dumps([h for (h,_s) in accepted], ensure_ascii=False),
        "relevance_details": json.dumps(relevance_details, ensure_ascii=False),

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

        "pplx_provenance": json.dumps(pplx_items, ensure_ascii=False),
        "pplx_last_error": pplx_err or ""
    }

    run_id = save_run(payload)

    # 13) Export artifacts (JSON, bars CSV, accepted headlines TXT)
    try:
        export_run_json(run_id, payload)
        N = 240
        bars_to_save = bars.tail(N) if len(bars) > N else bars
        save_bars_csv(run_id, bars_to_save)
        save_accepted_txt(run_id, accepted_with_src)
    except Exception:
        pass

    # 14) Auto-commit
    try:
        paths_to_commit = ["runs", "bars"]
        result = auto_commit_and_push(paths_to_commit, extra_message=f"run_id={run_id}", push_enabled=True)
        print(f"\nGit: {result}")
    except Exception:
        pass

    # 15) Output
    print(f"Bars={len(bars)} Headlines={len(merged_heads)} (used {used_cnt}, dropped {drop_cnt})")

    print("\n[Accepted (source, score)]")
    for item in accepted_with_src[:10]:
        print(f"- {item['source']} | {item['relevance']:.3f} | {item['headline'][:120]}")

    # Optional: top-5 relevance preview for tuning
    all_scored = accepted + rejected
    top5 = sorted(all_scored, key=lambda x: x[1], reverse=True)[:5]
    print("\n[Relevance top-5]")
    for h, s in top5:
        print(f"- {s:.3f} | {h[:120]}")

    print("\n=== Decision Preview ===")
    print(f"Story: {story} | Confidence: {conf:.2f}  ({explain_term('Confidence')})")
    print(f"Narrative: raw={raw_narr:+.2f} decayed={dec_narr:+.2f} → {nar_lbl}  ({explain_term('Narrative')})")
    print(f"  FinBERT (relevant-only, robust): {fin:+.2f}  kept={fin_kept_cnt} dropped={fin_drop_cnt}  ({explain_term('FinBERT')})")
    print(f"Price: {px:+.2f} → {px_lbl}  ({explain_term('Price score')})")
    print(f"  Volume Z: {vz:+.2f} → {vol_lbl}  ({explain_term('Volume Z')})")
    print(f"Gap (story vs price): {div:+.2f} → {gap_lbl}  ({explain_term('Gap')})")
    print(f"Trigger (act if gap >): {trig:.2f}  ({explain_term('Threshold')})")
    print(f"Action: {action}  ({explain_term('Action')})")
    print(f"Reason: {rc}  ({explain_term('Reason')})")

    what_it_means = (
        f"In simple terms: we merged available sources and only used BTC-relevant headlines "
        f"(used {used_cnt}, dropped {drop_cnt}). The news mood is {nar_lbl}, and the price mood is {px_lbl}. "
        f"Trading activity looks {vol_lbl}. The gap between story and price is {gap_lbl}. "
        f"Given the adaptive trigger ({trig:.2f}) and confidence {conf:.2f}, the move is {action}."
    )
    print("\nWhat this means:")
    print(what_it_means)

    print(f"\nSaved run_id={run_id} to tracer.db")

if __name__ == "__main__":
    main()
