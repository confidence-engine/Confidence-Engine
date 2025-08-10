from cascade import detect_cascade


class Bars:
    def __init__(self, closes, vols):
        self.close = closes
        self.volume = vols


def test_hype_only_detected():
    accepted = [
        {"headline": "BTC soars on ETF news"},
        {"headline": "Bitcoin surges on ETF approval"},  # near-dup
        {"headline": "BTC skyrockets as ETF flows grow"},  # near-dup
        {"headline": "BTC skyrockets as ETF flows grow"},  # exact dup
    ]
    bars = Bars([100.0, 100.1], [10.0, 10.1])
    res = detect_cascade(accepted, bars)
    assert res["tag"] == "HYPE_ONLY"
    assert res["confidence_delta"] == -0.03


def test_low_repetition_no_flag():
    accepted = [
        {"headline": "BTC rises"},
        {"headline": "ETH falls"},
    ]
    bars = Bars([100.0, 101.0], [10.0, 12.0])
    res = detect_cascade(accepted, bars)
    assert res["tag"] == ""
    assert res["confidence_delta"] == 0.0


def test_edge_len_lt_2():
    accepted = [{"headline": "solo"}]
    bars = Bars([100.0], [10.0])
    res = detect_cascade(accepted, bars)
    assert res["repetition_ratio"] == 0.0

