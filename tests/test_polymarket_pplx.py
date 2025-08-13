import json
from typing import Dict, Any

from providers.polymarket_pplx import get_crypto_markets_via_pplx
from scripts.polymarket_bridge import discover_and_map


def fake_pplx_fetch(url: str, data: bytes, headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
    # Simulate Perplexity chat completion returning JSON array in first choice
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps([
                        {
                            "title": "Will BTC be above $100k by Dec 31, 2025?",
                            "endDate": "2025-12-31T23:59:59Z",
                            "liquidityUSD": 250000,
                            "impliedProbability": 0.42,
                            "resolutionSource": "Coinbase BTC-USD close"
                        },
                        {
                            "title": "Will ETH be above $10k by Dec 31, 2025?",
                            "endDate": "2025-12-31T23:59:59Z",
                            "liquidityUSD": 180000,
                            "impliedProbability": 0.28,
                            "resolutionSource": "Coinbase ETH-USD close"
                        }
                    ])
                }
            }
        ]
    }
    return payload


def test_pplx_provider_normalizes_items():
    items = get_crypto_markets_via_pplx(assets_env="BTC,ETH", limit=5, fetch=fake_pplx_fetch)
    assert isinstance(items, list) and len(items) == 2
    assert items[0]["title"].lower().startswith("will btc")
    assert items[0]["endDate"].startswith("2025-")
    assert "impliedProbability" in items[0]


def test_bridge_mapping_with_pplx(monkeypatch):
    # monkeypatch the selector to return our fake pplx items
    from scripts import polymarket_bridge as bridge

    def fake_get_markets_via_source(**kwargs):
        return get_crypto_markets_via_pplx(assets_env="BTC,ETH", limit=5, fetch=fake_pplx_fetch)

    monkeypatch.setattr(bridge, "_get_markets_via_source", fake_get_markets_via_source)

    mapped = discover_and_map(
        min_liquidity=0,
        min_weeks=0,
        max_weeks=200,
        max_items=5,
        min_quality=0.0,
        fetch_fn=None,
    )
    assert len(mapped) == 2
    for m in mapped:
        assert m["symbol"].startswith("POLY:")
        assert m["rationale_chat"] and isinstance(m["rationale_chat"], str)
        assert "implied_prob" in m
        assert "end_date_iso" in m
