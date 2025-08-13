import os
import unittest

from scripts.tg_digest_formatter import render_digest
from scripts.discord_formatter import digest_to_discord_embeds


class TestExecutiveLeadersMessaging(unittest.TestCase):
    def setUp(self):
        # Ensure deterministic env
        os.environ.pop("TB_DIGEST_HIDE_EQUITIES_IF_NO_DATA", None)

    def _base_inputs(self):
        weekly = {"regime": "mixed", "plan_text": "Plan."}
        engine = {"thesis_text": "Tape mixed; be selective."}
        assets_ordered = ["BTC/USD", "ETH/USD"]
        assets_data = {
            "BTC/USD": {"thesis": {"action": "Buy", "risk_band": "Medium", "readiness": "Now"}},
            "ETH/USD": {"thesis": {"action": "Buy", "risk_band": "Medium", "readiness": "Now"}},
        }
        return weekly, engine, assets_ordered, assets_data

    def test_tg_leaders_skew_long(self):
        weekly, engine, assets_ordered, assets_data = self._base_inputs()
        out = render_digest(
            timestamp_utc="2025-01-01T00:00:00Z",
            weekly=weekly,
            engine=engine,
            assets_ordered=assets_ordered,
            assets_data=assets_data,
            options={},
            polymarket=[],
        )
        self.assertIn("Leaders skew long; wait for clean triggers.", out)

    def test_discord_leaders_skew_long(self):
        weekly, engine, assets_ordered, assets_data = self._base_inputs()
        digest = {
            "timestamp": "2025-01-01T00:00:00Z",
            "executive_take": engine.get("thesis_text"),
            "weekly": weekly,
            "engine": engine,
            "assets": [
                {"symbol": "BTC/USD", "thesis": assets_data["BTC/USD"]["thesis"]},
                {"symbol": "ETH/USD", "thesis": assets_data["ETH/USD"]["thesis"]},
            ],
        }
        embeds = digest_to_discord_embeds(digest)
        self.assertTrue(any("Leaders skew long" in (e.get("description") or "") for e in embeds if "description" in e))


if __name__ == "__main__":
    unittest.main()
