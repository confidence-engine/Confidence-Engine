"""
Evaluation metrics for prediction tasks (binary outcomes):
- Brier score
- Log loss
- Calibration curve (binning)
- Cohort win-rates

Pure-Python, no external deps.
"""
from __future__ import annotations
from typing import Iterable, List, Tuple, Dict


def _clip_prob(p: float, eps: float = 1e-12) -> float:
    if p < eps:
        return eps
    if p > 1 - eps:
        return 1 - eps
    return p


def brier_score(probs: Iterable[float], outcomes: Iterable[int]) -> float:
    """Mean squared error between probabilities and binary outcomes {0,1}."""
    p = list(probs)
    y = list(outcomes)
    n = len(p)
    if n == 0 or n != len(y):
        raise ValueError("probs/outcomes must be non-empty and same length")
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / n


def log_loss(probs: Iterable[float], outcomes: Iterable[int], eps: float = 1e-12) -> float:
    """Negative log-likelihood (base e) for binary outcomes.
    Uses clipping to avoid log(0).
    """
    import math

    p = list(probs)
    y = list(outcomes)
    n = len(p)
    if n == 0 or n != len(y):
        raise ValueError("probs/outcomes must be non-empty and same length")
    total = 0.0
    for pi, yi in zip(p, y):
        pc = _clip_prob(float(pi), eps)
        if yi == 1:
            total += -math.log(pc)
        else:
            total += -math.log(1.0 - pc)
    return total / n


def calibration_curve(
    probs: Iterable[float],
    outcomes: Iterable[int],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Returns a list of bins with fields:
    - bin_low, bin_high: probability interval (exclusive high for last may be ==1)
    - count: number of samples in bin
    - avg_pred: mean predicted prob in bin
    - frac_pos: empirical fraction of positives in bin
    Bins are equal-width in probability space [0,1].
    """
    p = list(probs)
    y = list(outcomes)
    if len(p) == 0 or len(p) != len(y):
        raise ValueError("probs/outcomes must be non-empty and same length")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    bins = [
        {"bin_low": i / n_bins, "bin_high": (i + 1) / n_bins, "sum_p": 0.0, "sum_y": 0.0, "count": 0}
        for i in range(n_bins)
    ]

    # Assign to bins; edge-case p==1.0 placed in last bin
    for pi, yi in zip(p, y):
        idx = int(min(n_bins - 1, max(0, int(pi * n_bins))))
        b = bins[idx]
        b["sum_p"] += float(pi)
        b["sum_y"] += int(yi)
        b["count"] += 1

    out = []
    for b in bins:
        cnt = b["count"]
        if cnt == 0:
            avg_pred = 0.0
            frac_pos = 0.0
        else:
            avg_pred = b["sum_p"] / cnt
            frac_pos = b["sum_y"] / cnt
        out.append(
            {
                "bin_low": b["bin_low"],
                "bin_high": b["bin_high"],
                "count": cnt,
                "avg_pred": avg_pred,
                "frac_pos": frac_pos,
            }
        )
    return out


def cohort_win_rates(
    probs: Iterable[float],
    outcomes: Iterable[int],
    labels: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    """Compute win-rates per cohort label.
    Returns mapping label -> {count, wins, win_rate, avg_prob}.
    """
    from collections import defaultdict

    p = list(probs)
    y = list(outcomes)
    l = list(labels)
    if not (len(p) and len(p) == len(y) == len(l)):
        raise ValueError("probs/outcomes/labels must be non-empty and same length")

    agg = defaultdict(lambda: {"count": 0, "wins": 0, "sum_prob": 0.0})
    for pi, yi, li in zip(p, y, l):
        a = agg[str(li)]
        a["count"] += 1
        a["wins"] += int(yi)
        a["sum_prob"] += float(pi)

    out: Dict[str, Dict[str, float]] = {}
    for k, v in agg.items():
        cnt = v["count"]
        out[k] = {
            "count": float(cnt),
            "wins": float(v["wins"]),
            "win_rate": (v["wins"] / cnt) if cnt > 0 else 0.0,
            "avg_prob": (v["sum_prob"] / cnt) if cnt > 0 else 0.0,
        }
    return out
