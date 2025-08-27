import datetime as dt
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.underrated_scanner import (
    Project,
    filter_and_rank,
    generate_narratives,
    _discord_embed_for,
    _telegram_text_for,
    _already_alerted,
    _mark_alerted,
    enrich_projects,
)


def _p(name, **kw):
    return Project(name=name, **kw)


def test_filter_and_rank_basic():
    a = _p("Alpha", desc="Great use case with strong fundamentals.", market_cap_usd=5_000_000, liquidity_score=0.7)
    b = _p("Bravo", desc="", market_cap_usd=9_000_000, liquidity_score=0.3)
    c = _p("Charlie", desc="Solid", market_cap_usd=200_000_000)  # over threshold; should drop

    ranked = filter_and_rank([a, b, c])

    # c filtered out
    names = [p.name for p in ranked]
    assert "Charlie" not in names
    # a should score higher than b
    assert names[0] == "Alpha"


def test_generate_narratives_timeline():
    fast = _p("Fast", liquidity_score=0.85)
    med = _p("Medium", liquidity_score=0.55)
    slow = _p("Slow", liquidity_score=0.2)

    assert generate_narratives(fast)["timeline_months"] == "3–6 months"
    assert generate_narratives(med)["timeline_months"] == "6–9 months"
    assert generate_narratives(slow)["timeline_months"] == "9–12 months"


def test_embed_and_telegram_formatting():
    p = _p(
        "Zeta",
        ticker="Z",
        desc="Zeta is a protocol for X.",
        market_cap_usd=12_345_678,
        liquidity_score=0.6,
        links=["https://zeta.org"],
    )
    nar = generate_narratives(p)
    p.why_matters = nar["why_matters"]
    p.potential_impact = nar["potential_impact"]
    p.timeline_months = nar["timeline_months"]

    embed = _discord_embed_for(p)
    assert embed["title"].startswith("Zeta")
    names = [f["name"] for f in embed["fields"]]
    assert "Market cap" in names
    assert "Liquidity score" in names

    text = _telegram_text_for(p)
    assert "Zeta" in text
    assert "Market cap:" in text


def test_dedupe_store_behavior():
    store = {"alerted": {}}
    p = _p("Omega", ticker="OMG")
    assert not _already_alerted(store, p)
    _mark_alerted(store, p)
    assert _already_alerted(store, p)
    # ISO timestamp stored
    ts = store["alerted"]["omega|omg"]
    dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))  # should not raise


@pytest.mark.parametrize("cg", [
    (100_000_000, 0.7),
])
@patch("scripts.underrated_scanner._coingecko_lookup")
def test_enrich_projects_uses_helpers(mock_cg, cg):
    mock_cg.return_value = cg

    p = _p("Theta", ticker="TH", links=["https://example.com"])
    out = enrich_projects([p])
    q = out[0]
    assert q.market_cap_usd == cg[0]
    assert q.liquidity_score == cg[1]


def test_main_interval_guard_skips(capsys):
    from scripts import underrated_scanner as und

    with patch.object(und, "_load_store", return_value={"last_run": (dt.datetime.utcnow().isoformat() + "Z")}):
        und.main()
        out = capsys.readouterr().out
        assert "Skipping run due to interval guard" in out


def test_main_happy_path_calls_senders(tmp_path, monkeypatch):
    from scripts import underrated_scanner as und

    # Make store old enough
    store = {"alerted": {}, "last_run": (dt.datetime.utcnow() - dt.timedelta(days=30)).isoformat() + "Z"}
    monkeypatch.setattr(und, "_load_store", lambda: store)
    monkeypatch.setattr(und, "_save_store", lambda s: None)
    monkeypatch.setattr(und, "save_report", lambda ranked, fresh: None)

    # Provide deterministic pipeline
    proj = Project(name="Phi", ticker="PHI", desc="Useful thing.")
    monkeypatch.setattr(und, "fetch_candidates", lambda: [proj])

    def enrich(ps):
        p = ps[0]
        p.market_cap_usd = 9_000_000
        p.liquidity_score = 0.6
        return ps

    monkeypatch.setattr(und, "enrich_projects", enrich)
    monkeypatch.setattr(und, "filter_and_rank", lambda ps: ps)

    sent = SimpleNamespace(discord=0, telegram=0)

    def fake_send(new_projects):
        sent.discord = 1
        sent.telegram = len(new_projects)
        return 1, len(new_projects)

    monkeypatch.setattr(und, "format_and_send_alerts", fake_send)

    und.main()

    assert sent.discord == 1
    assert sent.telegram == 1
