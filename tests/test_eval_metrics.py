from scripts.eval_metrics import brier_score, log_loss, calibration_curve, cohort_win_rates
import math


def test_brier_score_basic():
    probs = [0.9, 0.8, 0.2, 0.1]
    outs = [1, 1, 0, 0]
    # perfect-ish
    v = brier_score(probs, outs)
    assert 0 <= v < 0.05


def test_log_loss_clipping():
    probs = [0.0, 1.0]
    outs = [0, 1]
    v = log_loss(probs, outs)
    assert v < 1e-6  # near zero after clipping


def test_calibration_curve_bins():
    probs = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    outs =  [0,    0,    0,    0,    0,    1,    1,    1,    1,    1   ]
    bins = calibration_curve(probs, outs, n_bins=10)
    assert len(bins) == 10
    # ensure counts distributed
    assert sum(b["count"] for b in bins) == len(probs)


def test_cohort_win_rates():
    probs = [0.8, 0.7, 0.6, 0.4]
    outs = [1, 1, 1, 0]
    labels = ["A", "A", "B", "B"]
    res = cohort_win_rates(probs, outs, labels)
    assert set(res.keys()) == {"A", "B"}
    assert math.isclose(res["A"]["win_rate"], 1.0)
    assert math.isclose(res["B"]["win_rate"], 0.5)
