"""
Evidence line generator for surfaced assets.

Rules:
- Inputs: sentiment tag, participation tag, TF alignment boolean, signal quality tag, narrative tags.
- Output: 1–2 sentences, plain English, no numeric values.
- Provide a helper to strip numbers for chat outputs (TG/Discord), keeping artifacts untouched.
"""
from typing import List
import re


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
        return "Price action is bullish"
    if "bear" in s or s in {"sell", "short"}:
        return "Price action is bearish"
    if "watch" in s or "neutral" in s:
        return "Price action is mixed"
    return "Price action is developing"


def _choose_confidence_phrase(signal_quality: str, tf_aligned: bool) -> str:
    q = (signal_quality or "").strip().lower()
    if tf_aligned and ("strong" in q or "high" in q):
        return "with high confidence"
    if tf_aligned:
        return "with moderate confidence"
    if "weak" in q or "low" in q:
        return "with low confidence"
    return "with mixed confidence"


def _choose_participation_phrase(participation: str) -> str:
    p = (participation or "").strip().lower()
    if "hot" in p or "elevated" in p or "active" in p:
        return "Market participation is hot"
    if "quiet" in p or "thin" in p:
        return "Market participation is quiet"
    return "Market participation is normal"


def _choose_narrative_phrase(narrative_tags: List[str]) -> str:
    tags = " ".join((narrative_tags or [])).lower()
    if "continuation" in tags or "trend" in tags:
        return "and structure supports continuation"
    if "reversion" in tags or "fade" in tags:
        return "and structure favors mean reversion"
    if "breakout" in tags or "breakdown" in tags:
        return "and structure favors a breakout-based approach"
    return "and structure is defined"


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
