import re
from datetime import datetime, timedelta, timezone

from providers.polymarket import get_btc_eth_markets
from scripts.polymarket_bridge import discover_and_map


def _mk_market(title: str, weeks_from_now: int, liquidity: float, implied_prob: float = 0.6, quality: float = 0.9):
    end = (datetime.now(timezone.utc) + timedelta(weeks=weeks_from_now)).isoformat()
    return {
        "title": title,
        "resolutionSource": "CoinDesk",
        "endDate": end,
        "liquidity": liquidity,
        "impliedProbability": implied_prob,
        "quality": quality,
    }


def _fetch_ok(_url):
    return [
        _mk_market("Will Bitcoin close above X by month-end?", 2, 5000.0, 0.55, 0.95),
        _mk_market("Will Ethereum flip Bitcoin this quarter?", 8, 3000.0, 0.35, 0.80),
        _mk_market("Random sports market", 3, 10000.0, 0.50, 0.99),  # should be filtered out (not BTC/ETH)
        _mk_market("Bitcoin event next year", 60, 20000.0, 0.5, 0.9),  # end date too far
        {
            "title": "ETH question with no res source",
            "endDate": (datetime.now(timezone.utc) + timedelta(weeks=4)).isoformat(),
            "liquidity": 5000.0,
            "impliedProbability": 0.4,
        },  # missing resolutionSource
    ]


def test_provider_filters_btc_eth_resolution_time_liquidity():
    items = get_btc_eth_markets(
        min_liquidity=1000.0,
        min_weeks=1,
        max_weeks=12,
        max_items=None,
        fetch=_fetch_ok,
    )
    # Expect only the first two BTC/ETH items, both within window and with resolution source
    titles = [i["title"] for i in items]
    assert any("Bitcoin" in t for t in titles)
    assert any("Ethereum" in t or "ETH" in t for t in titles)
    assert all("Random sports" not in t for t in titles)
    assert all("next year" not in t for t in titles)
    assert all("no res source" not in t for t in titles)


def test_bridge_mapping_and_rationale_numeric_gating():
    mapped = discover_and_map(
        min_liquidity=1000.0,
        min_weeks=1,
        max_weeks=12,
        max_items=2,
        min_quality=0.5,
        fetch_fn=_fetch_ok,
    )
    assert len(mapped) <= 2
    for item in mapped:
        # Required chat fields
        assert item["symbol"].startswith("POLY:")
        assert item["readiness"] in ("Now", "Near", "Later")
        assert item["stance"] in ("Engage", "Stalk", "Stand Aside")
        assert item["edge_label"] in ("market cheap", "market rich", "in-line")

        # Rationale must be number-free
        rationale = item.get("rationale_chat") or ""
        assert isinstance(rationale, str) and len(rationale) > 0
        assert not re.search(r"\d", rationale), f"Rationale contains digits: {rationale}"

        # Artifact numeric fields must exist
        assert item.get("implied_prob") is not None
        assert item.get("internal_prob") is not None
        # end_date can be None but often present
