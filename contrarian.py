from __future__ import annotations


def decide_contrarian_viewport(narr_mag: float, divergence: float, price_move_pct: float) -> str:
    """
    Decide whether to surface a contrarian viewport tag indicating potential crowd error.

    Conditions (all must hold):
      - abs(narr_mag) >= 0.8  (extreme narrative magnitude)
      - abs(divergence) < 0.3 (low story-price gap)
      - abs(price_move_pct) < 0.2 (flat price over lookback)

    Returns "POTENTIAL_CROWD_MISTAKE" or "".
    """
    try:
        nm = abs(float(narr_mag))
        dv = abs(float(divergence))
        pm = abs(float(price_move_pct))
    except Exception:
        return ""

    if nm >= 0.8 and dv < 0.3 and pm < 0.2:
        return "POTENTIAL_CROWD_MISTAKE"
    return ""


