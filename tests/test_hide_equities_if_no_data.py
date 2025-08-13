import os
import unittest

from scripts.tg_digest_formatter import render_digest
from scripts.discord_formatter import digest_to_discord_embeds


class TestHideEquitiesIfNoData(unittest.TestCase):
    def setUp(self):
        os.environ["TB_DIGEST_HIDE_EQUITIES_IF_NO_DATA"] = "1"

    def test_tg_hides_equities_without_spot(self):
        weekly = {"regime": "mixed"}
        engine = {"thesis_text": "Mixed."}
        assets_ordered = ["AAPL"]
        assets_data = {
            "AAPL": {"symbol": "AAPL", "spot": None, "thesis": {"risk_band": "Medium", "readiness": "Later", "action": "Watch"}}
        }
        out = render_digest(
            timestamp_utc="2025-01-01T00:00:00Z",
            weekly=weekly,
            engine=engine,
            assets_ordered=assets_ordered,
            assets_data=assets_data,
            options={},
            polymarket=[],
        )
        # AAPL header should not appear
        self.assertNotIn("AAPL —", out)

    def test_discord_hides_equities_without_spot(self):
        digest = {
            "timestamp": "2025-01-01T00:00:00Z",
            "executive_take": "Mixed.",
            "weekly": {"regime": "mixed"},
            "engine": {"thesis_text": "Mixed."},
            "assets": [
                {"symbol": "AAPL", "spot": None, "thesis": {"risk_band": "Medium", "readiness": "Later", "action": "Watch"}},
            ],
        }
        embeds = digest_to_discord_embeds(digest)
        # Ensure there is at least one embed
        self.assertTrue(len(embeds) > 0)
        # Assets appear as separate embeds/fields after header and polymarket; ensure AAPL field isn't present
        assets_text = "\n".join("\n".join(f.get("value", "") for f in e.get("fields", [])) for e in embeds)
        self.assertNotIn("AAPL —", assets_text)


if __name__ == "__main__":
    unittest.main()
