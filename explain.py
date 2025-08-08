def strength_label(x: float) -> str:
    if x >= 0.75: return "very strong positive"
    if x >= 0.40: return "strong positive"
    if x >= 0.15: return "mild positive"
    if x > -0.15: return "neutral"
    if x > -0.40: return "mild negative"
    if x > -0.75: return "strong negative"
    return "very strong negative"

def volume_label(z: float) -> str:
    if z >= 1.0: return "high participation"
    if z >= 0.3: return "above-average participation"
    if z > -0.3: return "typical participation"
    if z > -0.8: return "below-average participation"
    return "weak participation"

def divergence_label(d: float, thresh: float) -> str:
    absd = abs(d)
    if absd >= thresh:
        return "large gap between story and price"
    if absd >= 0.5 * thresh:
        return "moderate gap between story and price"
    return "small gap between story and price"

def explain_term(term: str) -> str:
    gl = {
        "Narrative": "the mood from news headlines (story side).",
        "FinBERT": "an AI score of headline mood (positive/negative).",
        "Price score": "a combined view of indicators (RSI/MACD/trend).",
        "Volume Z": "how active trading is vs recent average.",
        "Gap": "difference between story mood and price mood.",
        "Threshold": "how big the gap must be before we act.",
        "Confidence": "how sure we are about the story signal.",
        "Action": "what we’d do now (buy/hold).",
        "Reason": "why we chose that action."
    }
    return gl.get(term, "")
