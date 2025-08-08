def compute(narrative_score: float, price_score: float) -> float:
    return narrative_score - price_score

def reason(divergence_value: float, conf: float, vol_z: float, thresh: float, conf_cut: float) -> str:
    if conf < conf_cut:
        return "LOW_CONFIDENCE"
    if vol_z < -0.5:
        return "WEAK_VOLUME"
    if abs(divergence_value) < thresh:
        return "SMALL_DIVERGENCE"
    return "NARR_LEADS_PRICE" if divergence_value > 0 else "PRICE_LEADS_NARR"
