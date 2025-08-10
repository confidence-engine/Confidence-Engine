from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple


def _tail_series(seq, n: int) -> List[float]:
    try:
        # pandas Series
        length = len(seq)
        return [float(seq.iloc[i]) for i in range(max(0, length - n), length)]
    except Exception:
        try:
            # list-like
            return [float(x) for x in list(seq)[-n:]]
        except Exception:
            return []


def _extract_tail(bars, n: int) -> Tuple[List[float], List[float]]:
    try:
        # pandas DataFrame path
        tail = bars.tail(n)
        closes = _tail_series(tail["close"], n)
        vols = _tail_series(tail["volume"], n)
        return closes, vols
    except Exception:
        # Fallback for simple objects with .close/.volume lists
        closes = _tail_series(getattr(bars, "close", []), n)
        vols = _tail_series(getattr(bars, "volume", []), n)
        return closes, vols


def _price_move_pct(closes: List[float]) -> Tuple[float, float]:
    if len(closes) < 2 or closes[0] == 0:
        return 0.0, 0.0
    signed = (closes[-1] - closes[0]) / closes[0] * 100.0
    return abs(signed), signed


def _volume_z(vols: List[float]) -> float:
    if len(vols) < 3:
        return 0.0
    mean = sum(vols) / float(len(vols))
    var = sum((v - mean) ** 2 for v in vols) / float(len(vols))
    std = var ** 0.5
    if std == 0:
        return 0.0
    last = vols[-1]
    return float((last - mean) / std)


def _price_score_from_signed_pct(signed_pct: float, k: float = 3.0) -> float:
    # tanh(k * signed_pct) as specified (percent units)
    return math.tanh(k * signed_pct)


def _get_weights() -> Dict[str, float]:
    ws = {
        "short": float(os.getenv("TB_TS_W_SHORT", "0.5") or 0.5),
        "mid": float(os.getenv("TB_TS_W_MID", "0.35") or 0.35),
        "long": float(os.getenv("TB_TS_W_LONG", "0.15") or 0.15),
    }
    total = sum(v for v in ws.values() if v >= 0)
    if total <= 0:
        return {"short": 0.5, "mid": 0.35, "long": 0.15}
    return {k: v / total for k, v in ws.items()}


def compute_timescale_scores(bars, narrative_sentiment: float) -> Dict:
    """
    Compute three-horizon divergence and a weighted aggregate.

    Notes:
    - narrative_sentiment is reused across horizons as a proxy for now (to be refined later).
    - Assumes bars are 1-minute; slicing uses tail(N).
    """
    horizons = {"short": 60, "mid": 180, "long": 360}
    out: Dict[str, Dict] = {}

    for name, n in horizons.items():
        closes, vols = _extract_tail(bars, n)
        pm_abs, pm_signed = _price_move_pct(closes)
        vz = _volume_z(vols)
        price_score = _price_score_from_signed_pct(pm_signed)
        divergence_h = float(narrative_sentiment) - float(price_score)
        out[name] = {
            "divergence": float(divergence_h),
            "price_move_pct": float(round(pm_abs, 3)),
            "volume_z": float(round(vz, 3)),
        }

    weights = _get_weights()
    combined = (
        weights["short"] * out["short"]["divergence"]
        + weights["mid"] * out["mid"]["divergence"]
        + weights["long"] * out["long"]["divergence"]
    )
    combined_sign = 1 if combined > 0 else (-1 if combined < 0 else 0)
    aligned = 0
    for name in ("short", "mid", "long"):
        d = out[name]["divergence"]
        if abs(d) >= 0.20 and (1 if d > 0 else (-1 if d < 0 else 0)) == combined_sign and combined_sign != 0:
            aligned += 1

    result = {
        **out,
        "combined_divergence": float(round(combined, 3)),
        "aligned_horizons": int(aligned),
        "alignment_flag": bool(aligned >= 2),
        "weights": {k: float(round(v, 3)) for k, v in weights.items()},
    }
    return result


