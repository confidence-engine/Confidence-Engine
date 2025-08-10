from contrarian import decide_contrarian_viewport


def test_contrarian_positive_case():
    tag = decide_contrarian_viewport(0.85, 0.25, 0.15)
    assert tag == "POTENTIAL_CROWD_MISTAKE"


def test_contrarian_negative_narrative():
    tag = decide_contrarian_viewport(0.75, 0.25, 0.15)
    assert tag == ""


def test_contrarian_negative_divergence():
    tag = decide_contrarian_viewport(0.85, 0.35, 0.15)
    assert tag == ""


def test_contrarian_negative_price_move():
    tag = decide_contrarian_viewport(0.85, 0.25, 0.25)
    assert tag == ""


