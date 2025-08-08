from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError, field_validator
from relevance import rank_relevance

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

def filter_relevant(headlines: List[str], threshold: float = 0.50) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    return rank_relevance(headlines, BTC_TOPIC, threshold=threshold)

def make_from_headlines(headlines: List[str], threshold: float = 0.50) -> Optional[Narrative]:
    if not headlines:
        return None
    accepted, _ = filter_relevant(headlines, threshold)
    use = [h for (h, _s) in accepted]
    if not use:
        return None
    summary = use[0][:256]
    return Narrative(
        narrative_summary=summary,
        narrative_momentum_score=0.0,
        confidence=0.65 if len(use) >= 2 else 0.55,
        salient_entities=[],
        anchor_quotes=[summary[:120]]
    )
