from typing import Dict, Optional

# Central place to tune source weights for relevance
# Rationale:
# - Perplexity returns curated crypto/BTC news → weight 1.10
# - CoinDesk is crypto-native → weight 1.05
# - Alpaca may include broad market headlines → weight 1.00
# Adjust as you learn quality over time.

DEFAULT_SOURCE_WEIGHTS = {
    "perplexity": 1.10,
    "coindesk": 1.05,
    "alpaca": 1.00,
    "unknown": 1.00,
}

def get_source_weight(src: str, overrides: Optional[Dict[str, float]] = None) -> float:
    m = DEFAULT_SOURCE_WEIGHTS.copy()
    if overrides:
        m.update({k: float(v) for k, v in overrides.items()})
    return float(m.get(src, 1.00))
