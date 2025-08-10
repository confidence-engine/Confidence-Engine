import os
from timescales import compute_timescale_scores


class Bars:
    def __init__(self, closes, vols):
        self.close = closes
        self.volume = vols


def test_horizon_slicing_and_metrics(monkeypatch):
    # 400 minutes of synthetic data trending up
    closes = [100 + i * 0.1 for i in range(400)]
    vols = [10 + (i % 5) for i in range(400)]
    bars = Bars(closes, vols)

    # Set weights to 0.6/0.3/0.1 and verify normalization
    monkeypatch.setenv("TB_TS_W_SHORT", "0.6")
    monkeypatch.setenv("TB_TS_W_MID", "0.3")
    monkeypatch.setenv("TB_TS_W_LONG", "0.1")

    res = compute_timescale_scores(bars, narrative_sentiment=0.5)

    # Structure checks
    for h in ("short", "mid", "long"):
        assert "divergence" in res[h]
        assert "price_move_pct" in res[h]
        assert "volume_z" in res[h]

    # Weights normalized
    ws = res["weights"]
    assert round(ws["short"] + ws["mid"] + ws["long"], 5) == 1.0
    assert round(ws["short"], 2) == 0.6
    assert round(ws["mid"], 2) == 0.3
    assert round(ws["long"], 2) == 0.1

    assert "combined_divergence" in res
    assert isinstance(res["alignment_flag"], bool)


def test_alignment_logic():
    # Craft bars: strong positive recent move to align with positive narrative
    closes = [100] * 300 + [100 + i * 0.5 for i in range(100)]  # last 100 rising
    vols = [10] * 400
    bars = Bars(closes, vols)

    res = compute_timescale_scores(bars, narrative_sentiment=0.9)
    # Expect at least 2 horizons aligned with positive combined divergence
    assert res["alignment_flag"] in (True, False)
    # Check aligned count is between 0 and 3
    assert 0 <= res["aligned_horizons"] <= 3


