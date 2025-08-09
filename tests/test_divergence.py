from divergence import compute, reason
import pytest


def test_compute_simple():
    assert compute(0.7, 0.2) == pytest.approx(0.5, rel=1e-12, abs=1e-12)
    assert compute(-0.1, 0.2) == pytest.approx(-0.3, rel=1e-12, abs=1e-12)


def test_reason_paths():
    # LOW_CONFIDENCE
    assert reason(2.0, conf=0.4, vol_z=0.0, thresh=1.0, conf_cut=0.6) == "LOW_CONFIDENCE"
    # WEAK_VOLUME
    assert reason(2.0, conf=0.7, vol_z=-0.6, thresh=1.0, conf_cut=0.6) == "WEAK_VOLUME"
    # SMALL_DIVERGENCE
    assert reason(0.5, conf=0.7, vol_z=0.0, thresh=1.0, conf_cut=0.6) == "SMALL_DIVERGENCE"
    # NARR_LEADS_PRICE vs PRICE_LEADS_NARR
    assert reason(1.5, conf=0.8, vol_z=0.2, thresh=1.0, conf_cut=0.6) == "NARR_LEADS_PRICE"
    assert reason(-1.5, conf=0.8, vol_z=0.2, thresh=1.0, conf_cut=0.6) == "PRICE_LEADS_NARR"


