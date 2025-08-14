"""
Evidence line generator for surfaced assets.

Rules:
- Inputs: sentiment tag, participation tag, TF alignment boolean, signal quality tag, narrative tags.
- Output: 1–2 sentences, plain English, no numeric values.
- Provide a helper to strip numbers for chat outputs (TG/Discord), keeping artifacts untouched.
"""
from typing import List
import re

def estimate_confidence_pct(signal_quality_tag: str, tf_aligned: bool, risk_band: str) -> float:
    """
    Lightweight agent confidence estimator in [0.0, 1.0].
    Rules of thumb:
    - signal_quality: low≈0.4, moderate≈0.55, strong/high≈0.7, very high≈0.85
    - +0.10 if tf_aligned
    - Risk penalty: risk_band high -> -0.05, extreme -> -0.10
    - Clamp to [0.2, 0.98]
    """
    sq = (signal_quality_tag or "").strip().lower()
    if ("very" in sq and "high" in sq) or ("vh" in sq):
        base = 0.85
    elif "strong" in sq or "high" in sq:
        base = 0.70
    elif "mod" in sq or "medium" in sq or "moderate" in sq:
        base = 0.55
    elif "weak" in sq or "low" in sq:
        base = 0.40
    else:
        base = 0.50
    if bool(tf_aligned):
        base += 0.10
    rb = (risk_band or "").strip().lower()
    if "extreme" in rb:
        base -= 0.10
    elif "high" in rb:
        base -= 0.05
    return max(0.20, min(0.98, base))


def generate_high_risk_note(risk_band: str, action: str, risk_score: float) -> str:
    """
    Generate a graded risk explanation when risk is high and action is Buy/Watch.
    Returns empty string otherwise.
    """
    rb = (risk_band or "").strip().lower()
    act = (action or "").strip().title()
    try:
        rs = float(risk_score)
    except Exception:
        rs = 0.0
    if rb != "high" or act not in ("Buy", "Watch"):
        return ""
    if rs >= 0.85:
        level = "extreme"
    elif rs >= 0.7:
        level = "very high"
    else:
        level = "elevated"
    if act == "Buy":
        return f"{level.capitalize()}-risk long setup; size aggressively low and expect wide swings."
    else:
        return f"{level.capitalize()}-risk potential; only suitable if comfortable with elevated volatility."


def strip_numbers_for_chat(text: str) -> str:
    """
    Remove numeric values from a string for safe chat rendering.
    Preserves letters and punctuation, collapses extra whitespace.
    """
    if not text:
        return ""
    # Remove digits and common numeric formats (including decimals, percents, +/-, commas)
    cleaned = re.sub(r"[\d]+(?:[\.,][\d]+)?%?", "", text)
    # Remove stray plus/minus signs attached to numbers we just stripped
    cleaned = re.sub(r"[\+\-]\s*", "", cleaned)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _choose_sentiment_phrase(sentiment: str) -> str:
    s = (sentiment or "").strip().lower()
    if "bull" in s or s in {"buy", "long"}:
        return "Price looks bullish"
    if "bear" in s or s in {"sell", "short"}:
        return "Price looks bearish"
    if "watch" in s or "neutral" in s:
        return "Price looks mixed"
    return "Price is developing"


def _choose_confidence_phrase(signal_quality: str, tf_aligned: bool) -> str:
    q = (signal_quality or "").strip().lower()
    very_high = ("very" in q and "high" in q) or ("vh" in q)
    high = ("strong" in q or "high" in q)
    if very_high and tf_aligned:
        return "with very high confidence"
    if very_high and not tf_aligned:
        # Dominant TF leads; minor divergences present
        return "with very high confidence; dominant timeframe leads; minor divergences present"
    if tf_aligned and high:
        return "with high confidence"
    if tf_aligned:
        return "with moderate confidence"
    if "weak" in q or "low" in q:
        return "with low confidence"
    return "with mixed confidence"


def _choose_participation_phrase(participation: str) -> str:
    p = (participation or "").strip().lower()
    if "hot" in p or "elevated" in p or "active" in p:
        return "Trading activity is busy"
    if "quiet" in p or "thin" in p:
        return "Trading activity is quiet"
    return "Trading activity is normal"


def _choose_narrative_phrase(narrative_tags: List[str]) -> str:
    tags = " ".join((narrative_tags or [])).lower()
    if "continuation" in tags or "trend" in tags:
        return "and the pattern supports continuation"
    if "reversion" in tags or "fade" in tags:
        return "and the pattern favors mean reversion"
    if "breakout" in tags or "breakdown" in tags:
        return "and the pattern favors breakouts"
    return "and the pattern is well-defined"


def generate_evidence_line(
    sentiment_tag: str,
    participation_tag: str,
    tf_aligned: bool,
    signal_quality_tag: str,
    narrative_tags: List[str],
) -> str:
    """
    Generate a concise, number‑free "why now" string.

    Returns up to 2 sentences, no numeric values.
    """
    # Sentence 1: Sentiment + confidence
    s1 = f"{_choose_sentiment_phrase(sentiment_tag)} {_choose_confidence_phrase(signal_quality_tag, tf_aligned)}."
    # Sentence 2: Participation + narrative
    s2 = f"{_choose_participation_phrase(participation_tag)}, {_choose_narrative_phrase(narrative_tags)}."

    out = f"{s1} {s2}".strip()
    # Ensure no numbers, even if user passed numeric-like tags
    out = strip_numbers_for_chat(out)
    return out
