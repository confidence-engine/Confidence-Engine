from diversity import compute_diversity_confidence_adjustment


def test_echo_chamber_penalty():
    # 4 out of 5 from same source (80%) → penalty -0.02 (boost +0.03 for >=2 sources)
    accepted = (
        [{"source": "perplexity"}] * 4
        + [{"source": "alpaca"}] * 1
    )
    adj, meta = compute_diversity_confidence_adjustment(accepted)
    assert meta["unique"] == 2
    # boost 0.03 + penalty -0.02 = 0.01
    assert round(adj, 2) == 0.01


def test_two_sources_boost():
    accepted = [
        {"source": "perplexity"},
        {"source": "alpaca"},
    ]
    adj, meta = compute_diversity_confidence_adjustment(accepted)
    assert meta["unique"] == 2
    assert round(adj, 2) == 0.03


def test_three_sources_boost():
    accepted = [
        {"source": "perplexity"},
        {"source": "alpaca"},
        {"source": "coindesk"},
    ]
    adj, meta = compute_diversity_confidence_adjustment(accepted)
    assert meta["unique"] == 3
    assert round(adj, 2) == 0.05


def test_empty_is_zero():
    adj, meta = compute_diversity_confidence_adjustment([])
    assert adj == 0.0
    assert meta["unique"] == 0
    assert meta["top_source_share"] == 0.0
    assert meta["counts"] == {}


