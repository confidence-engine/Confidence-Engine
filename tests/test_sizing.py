import os
from sizing import map_confidence_to_R


def test_below_floor_zero():
    os.environ.pop("TB_SIZE_CONF_FLOOR", None)
    res = map_confidence_to_R(0.60)
    assert res["target_R"] == 0.0
    assert "below floor" in res["notes"]


def test_linear_scaling_and_caps(monkeypatch):
    monkeypatch.setenv("TB_SIZE_CONF_FLOOR", "0.60")
    monkeypatch.setenv("TB_SIZE_CONF_CAP", "0.80")
    monkeypatch.setenv("TB_SIZE_MIN_R", "0.20")
    monkeypatch.setenv("TB_SIZE_MAX_R", "1.00")
    # Midpoint confidence → midpoint R (0.60..0.80 maps to 0.20..1.00)
    res = map_confidence_to_R(0.70)
    assert round(res["target_R"], 2) == 0.60


def test_vol_normalization(monkeypatch):
    monkeypatch.setenv("TB_SIZE_CONF_FLOOR", "0.60")
    monkeypatch.setenv("TB_SIZE_CONF_CAP", "0.80")
    monkeypatch.setenv("TB_SIZE_MIN_R", "0.20")
    monkeypatch.setenv("TB_SIZE_MAX_R", "1.00")
    res = map_confidence_to_R(0.80, vol_norm=2.0)
    # Without vol norm, would be 1.00; with vol norm=2.0, becomes 0.50
    assert round(res["target_R"], 2) == 0.50
    assert "vol_normalized" in res["notes"]


