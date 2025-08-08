import math

def blend(llm_score: float, finbert_score: float, w_llm: float = 0.6, w_fin: float = 0.4) -> float:
    s = w_llm * llm_score + w_fin * finbert_score
    return max(-1.0, min(1.0, s))

def decay(score: float, minutes_since: float, half_life_min: float = 90.0) -> float:
    if minutes_since is None or minutes_since < 0:
        return score
    lam = math.log(2) / max(1e-6, half_life_min)
    return score * math.exp(-lam * minutes_since)
