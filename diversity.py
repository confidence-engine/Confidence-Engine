from __future__ import annotations

from typing import Dict, List, Tuple


def compute_diversity_confidence_adjustment(
    accepted: List[Dict]
) -> Tuple[float, Dict]:
    """
    Compute a confidence adjustment based on source diversity among accepted headlines.

    Rules:
    - unique_sources = count of distinct 'source' in accepted
    - boost: +0.03 if unique_sources>=2; +0.05 if >=3 (cap boost at +0.05)
    - echo chamber penalty: top_source_share = max(count_by_source)/len(accepted) if accepted>0.
        If top_source_share>=0.70 apply −0.02; if >=0.85 apply −0.05.
    - net adjustment = boost + penalty; clamp to [-0.05, +0.05].
    - cap final confidence elsewhere; this function returns only the adjustment.

    Returns (adjustment, meta) where
      meta = {"unique": N, "top_source_share": float, "counts": {source: count}}
    """
    counts: Dict[str, int] = {}
    for item in accepted or []:
        src = (item.get("source") or "unknown").strip().lower()
        counts[src] = counts.get(src, 0) + 1

    total = sum(counts.values())
    unique_sources = len(counts)

    # Boost
    boost = 0.0
    if unique_sources >= 3:
        boost = 0.05
    elif unique_sources >= 2:
        boost = 0.03

    # Penalty
    top_share = 0.0
    if total > 0:
        top_cnt = max(counts.values())
        top_share = top_cnt / float(total)
    penalty = 0.0
    if top_share >= 0.85:
        penalty = -0.05
    elif top_share >= 0.75:
        penalty = -0.03

    adj = boost + penalty
    if adj > 0.05:
        adj = 0.05
    if adj < -0.05:
        adj = -0.05

    meta = {
        "unique": unique_sources,
        "top_source_share": round(top_share, 3),
        "counts": counts,
    }
    return float(adj), meta


