import os
import unittest

from scripts.tg_digest_formatter import render_digest
from scripts.discord_formatter import digest_to_discord_embeds


PM_SAMPLE = [
    {
        "title": "Bitcoin above 120k on August 13?",
        "readiness": "Now",
        "stance": "Engage",
        "edge_label": "market rich",
        "rationale_chat": "Market pricing looks stretched versus our view. Timing is actionable; manage risk and execution.",
        "outcome_label": "Lean YES",
        "implied_side": "YES",
        "implied_pct": 99,
        "implied_prob": 0.99,
        "internal_prob": 0.95,
    }
]


class TestPolymarketNumbersGate(unittest.TestCase):
    def setUp(self):
        # default off
        os.environ["TB_POLYMARKET_NUMBERS_IN_CHAT"] = "0"

    def test_tg_numbers_suppressed(self):
        out = render_digest(
            timestamp_utc="2025-01-01T00:00:00Z",
            weekly={"regime": "mixed"},
            engine={"thesis_text": "Mixed."},
            assets_ordered=[],
            assets_data={},
            options={},
            polymarket=PM_SAMPLE,
        )
        self.assertIn("Polymarket BTC/ETH", out)
        self.assertNotIn("Outcome:", out)

    def test_discord_numbers_suppressed(self):
        digest = {
            "timestamp": "2025-01-01T00:00:00Z",
            "executive_take": "Mixed.",
            "weekly": {"regime": "mixed"},
            "engine": {"thesis_text": "Mixed."},
            "assets": [],
            "polymarket": PM_SAMPLE,
        }
        embeds = digest_to_discord_embeds(digest)
        # Polymarket embed should not contain Outcome line
        poly = next((e for e in embeds if e.get("title") == "Polymarket BTC/ETH"), None)
        self.assertIsNotNone(poly)
        text = "\n".join(v.get("value", "") for v in poly.get("fields", []))
        self.assertNotIn("Outcome:", text)

    def test_enable_numbers_shows_outcome(self):
        os.environ["TB_POLYMARKET_NUMBERS_IN_CHAT"] = "1"
        out = render_digest(
            timestamp_utc="2025-01-01T00:00:00Z",
            weekly={"regime": "mixed"},
            engine={"thesis_text": "Mixed."},
            assets_ordered=[],
            assets_data={},
            options={"TB_POLYMARKET_SHOW_OUTCOME": "1"},
            polymarket=PM_SAMPLE,
        )
        self.assertIn("Outcome:", out)


if __name__ == "__main__":
    unittest.main()
