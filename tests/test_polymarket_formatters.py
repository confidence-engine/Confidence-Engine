from scripts.tg_digest_formatter import render_digest
from scripts.discord_formatter import digest_to_discord_embeds


def _minimal_assets():
    return {
        "BTC/USD": {
            "thesis": {"risk_band": "Medium", "readiness": "Later", "action": "Watch"},
            "structure": "trend / range context",
            "spot": None,
            "plan": {},
            "weekly_anchor": {},
            "sizing_text": "",
        }
    }


def test_tg_formatter_includes_polymarket_section():
    text = render_digest(
        timestamp_utc="2025-01-01T00:00:00Z",
        weekly={"regime": "mixed", "plan_text": "test plan", "anchors": []},
        engine={"thesis_text": "thesis", "evidence_bullets": []},
        assets_ordered=["BTC/USD"],
        assets_data=_minimal_assets(),
        options={"include_weekly": True, "include_engine": True, "include_prices": False},
        polymarket=[{
            "title": "Will Bitcoin close above X?",
            "stance": "Engage",
            "readiness": "Now",
            "edge_label": "market cheap",
            "rationale_chat": "Pricing looks favorable; timing is actionable.",
        }],
    )
    assert "Polymarket BTC/ETH" in text
    assert "Will Bitcoin close above X?" in text
    assert "Engage | Now | market cheap" in text
    assert "Pricing looks favorable" in text


def test_discord_formatter_includes_polymarket_embed():
    digest_data = {
        "timestamp": "2025-01-01T00:00:00Z",
        "executive_take": "exec",
        "weekly": {"regime": "mixed", "plan_text": "plan", "anchors": []},
        "engine": {"thesis_text": "thesis", "evidence_bullets": []},
        "assets": [{
            "symbol": "BTC/USD",
            "structure": "trend",
            "risk": "Medium",
            "readiness": "Later",
            "action": "Watch",
            "plan": {},
        }],
        "polymarket": [{
            "title": "Will Ethereum gain?",
            "stance": "Stalk",
            "readiness": "Near",
            "edge_label": "market rich",
            "rationale_chat": "Pricing looks stretched; timing is approaching.",
        }]
    }
    embeds = digest_to_discord_embeds(digest_data)
    # Find the polymarket embed
    pm = [e for e in embeds if e.get("title") == "Polymarket BTC/ETH"]
    assert pm, "Polymarket embed not found"
    fields = pm[0].get("fields") or []
    names = [f.get("name") for f in fields]
    assert any("Will Ethereum gain?" in n for n in names)
    values = [f.get("value") for f in fields]
    assert any("Stalk | Near | market rich" in v for v in values)
    assert any("Pricing looks stretched" in v for v in values)
