from typing import List, Optional, Tuple, Dict
from pydantic import BaseModel, Field, ValidationError, field_validator
from relevance import rank_relevance  # expected to return list[(headline, raw_score)]
from source_weights import get_source_weight

# Enriched topic improves BTC matching
BTC_TOPIC = "Bitcoin BTC BTC/USD price action, ETF flows, mining, halving, on-chain, Lightning, hash rate, SEC ETF, HODL, BTC dominance"

class Narrative(BaseModel):
    narrative_summary: str = Field(min_length=1, max_length=512)
    narrative_momentum_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    salient_entities: List[str] = Field(default_factory=list, max_length=5)
    anchor_quotes: List[str] = Field(default_factory=list, max_length=3)

    @field_validator("salient_entities", "anchor_quotes")
    @classmethod
    def _strip(cls, v: List[str]) -> List[str]:
        return [s.strip() for s in v if isinstance(s, str) and s.strip()]

def parse(payload: dict) -> Optional[Narrative]:
    try:
        return Narrative(**payload)
    except ValidationError:
        return None

def filter_relevant_weighted(
    headlines: List[str],
    threshold: float,
    source_lookup_fn,
    weight_overrides: Optional[Dict[str, float]] = None
) -> Tuple[List[Tuple[str, float, float, str]], List[Tuple[str, float, float, str]]]:
    """
    Returns:
      accepted: list of (headline, raw_score, weighted_score, source)
      rejected: same tuple
    source_lookup_fn(headline) -> 'perplexity'|'alpaca'|'coindesk'|'unknown'
    """
    acc, rej = rank_relevance(headlines, BTC_TOPIC, threshold=0.0)
    ranked: List[Tuple[str, float]] = acc + rej
    accepted, rejected = [], []
    for h, raw_s in ranked:
        src = source_lookup_fn(h)
        w = get_source_weight(src, overrides=weight_overrides)
        ws = raw_s * w
        target = accepted if ws >= threshold else rejected
        target.append((h, float(raw_s), float(ws), src))
    return accepted, rejected

def make_from_headlines(headlines: List[str], threshold: float = 0.50) -> Optional[Narrative]:
    if not headlines:
        return None
    summary = (headlines[0] if headlines else "")[:256]
    if not summary:
        return None
    conf = 0.65 if len(headlines) >= 2 else 0.55
    return Narrative(
        narrative_summary=summary,
        narrative_momentum_score=0.0,
        confidence=conf,
        salient_entities=[],
        anchor_quotes=[summary[:120]]
    )
