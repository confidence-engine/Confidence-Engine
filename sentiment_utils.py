from typing import List, Tuple
import numpy as np

def trimmed_mean(values: List[float], trim: float = 0.1) -> float:
    if not values:
        return 0.0
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    k = int(n * trim)
    if n <= 2 * k:
        return float(np.mean(v))
    return float(np.mean(v[k:n - k]))

def mad(a: np.ndarray) -> float:
    med = np.median(a)
    return float(np.median(np.abs(a - med)) + 1e-12)

def drop_outliers(values: List[float], z_thresh: float = 2.5) -> Tuple[List[float], List[float]]:
    """
    Robust outlier removal using MAD-based z-score.
    Returns (kept, dropped).
    """
    if not values:
        return [], []
    arr = np.asarray(values, dtype=float)
    med = np.median(arr)
    m = mad(arr)
    z = 0.6745 * (arr - med) / m
    keep_mask = np.abs(z) <= z_thresh
    kept = arr[keep_mask].tolist()
    dropped = arr[~keep_mask].tolist()
    return kept, dropped
