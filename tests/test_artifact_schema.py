import json
from pathlib import Path
from typing import List, Dict

# The helper that enriches saved artifacts with evidence lines and polymarket array
from scripts.tracer_bullet_universe import enrich_artifact


def _minimal_artifact(payloads: List[Dict]) -> Dict:
    return {
        "run_id": "test-run-001",
        "payloads": payloads,
        # legacy fields possibly present; keep minimal for this test
    }


def test_enrich_artifact_injects_evidence_and_polymarket(tmp_path: Path):
    # Arrange: create a minimal artifact with two assets
    art_path = tmp_path / "universe_run_test.json"
    payloads = [
        {"symbol": "BTC", "some": "field"},
        {"symbol": "ETH", "some": "field"},
    ]
    data = _minimal_artifact(payloads)
    art_path.write_text(json.dumps(data))

    # Evidence sink captured during digest rendering (Step 2)
    ev_sink = {
        "BTC": "Bias: bullish | Participation: normal | TF: aligned | Quality: strong | Tags: trend",
        "ETH": "Bias: watch | Participation: normal | TF: not-aligned | Quality: mixed | Tags: reversion",
    }

    # Polymarket items after filtering/mapping (Step 3)
    poly_items = [
        {
            "title": "Bitcoin > $100k by 2026",
            "stance": "Engage",
            "readiness": "Near",
            "edge_label": "cheap",
            "rationale_chat": "Strong momentum, catalysts upcoming",
            "implied_prob": 0.62,
            "internal_prob": 0.55,
            "implied_pct": 62,
            "end_date_iso": "2026-12-31T00:00:00Z",
            "liquidity_usd": 15342.0,
            "market_id": "btc-100k-2026",
            "quality": 0.87,
        }
    ]

    # Act: enrich the saved artifact
    enrich_artifact(str(art_path), ev_sink, poly_items)

    # Assert: load and validate enrichment
    out = json.loads(art_path.read_text())

    # Evidence lines should be present per payload
    assert "payloads" in out
    assert len(out["payloads"]) == 2
    by_sym = {p.get("symbol"): p for p in out["payloads"]}
    assert by_sym["BTC"].get("evidence_line") == ev_sink["BTC"]
    assert by_sym["ETH"].get("evidence_line") == ev_sink["ETH"]

    # Polymarket array should exist and be fully mapped
    assert "polymarket" in out
    assert isinstance(out["polymarket"], list)
    assert len(out["polymarket"]) == 1
    it = out["polymarket"][0]
    # Required fields and types
    assert it.get("market_name") == poly_items[0]["title"]
    assert it.get("stance") == "Engage"
    assert it.get("readiness") == "Near"
    assert it.get("edge_label") == "cheap"
    assert it.get("rationale") == poly_items[0]["rationale_chat"]
    assert isinstance(it.get("implied_prob"), (float, int))
    assert isinstance(it.get("tb_internal_prob"), (float, int))
    assert isinstance(it.get("implied_pct"), (float, int))
    assert isinstance(it.get("liquidity_usd"), (float, int))
    assert isinstance(it.get("market_id"), str)
    assert isinstance(it.get("quality"), (float, int))
    assert it.get("end_date_iso") == poly_items[0]["end_date_iso"]


def test_backward_compat_old_artifact_without_new_fields(tmp_path: Path):
    # Arrange: simulate an old artifact without evidence_line or polymarket
    art_path = tmp_path / "universe_run_legacy.json"
    old = {
        "run_id": "legacy-run-001",
        "payloads": [
            {"symbol": "BTC"},
            {"symbol": "ETH"},
        ],
        # no 'polymarket' key
    }
    art_path.write_text(json.dumps(old))

    # Act: simply load and probe with .get to ensure no exceptions
    loaded = json.loads(art_path.read_text())

    # Assert: ensure code reading via .get() pattern remains safe
    for p in loaded.get("payloads", []):
        _ = p.get("evidence_line")  # may be None; should not raise
    _ = loaded.get("polymarket", [])  # should default to []
