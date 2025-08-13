import os
import unittest

from scripts.evidence_lines import generate_high_risk_note
from scripts.tg_digest_formatter import render_digest
from scripts.discord_formatter import digest_to_discord_embeds


class TestHighRiskNotes(unittest.TestCase):
    def setUp(self):
        # Save env
        self._orig_flag = os.environ.get("TB_DIGEST_EXPLAIN_HIGH_RISK_ACTION")

    def tearDown(self):
        # Restore env
        if self._orig_flag is None:
            os.environ.pop("TB_DIGEST_EXPLAIN_HIGH_RISK_ACTION", None)
        else:
            os.environ["TB_DIGEST_EXPLAIN_HIGH_RISK_ACTION"] = self._orig_flag

    def test_helper_extreme_buy(self):
        note = generate_high_risk_note("high", "Buy", 0.9)
        self.assertIn("Extreme-risk long setup", note)

    def test_helper_very_high_watch(self):
        note = generate_high_risk_note("high", "Watch", 0.75)
        self.assertIn("Very high-risk potential", note)

    def test_helper_elevated_buy(self):
        note = generate_high_risk_note("high", "Buy", 0.68)
        self.assertIn("Elevated-risk long setup", note)

    def test_helper_non_high_no_note(self):
        self.assertEqual(generate_high_risk_note("medium", "Buy", 0.9), "")
        self.assertEqual(generate_high_risk_note("low", "Watch", 0.9), "")

    def _sample_assets(self):
        return {
            "BTC/USD": {
                "thesis": {
                    "risk_band": "high",
                    "readiness": "Now",
                    "action": "Buy",
                    "risk_score": 0.9,
                },
                "structure": "trend / range context",
                "spot": None,
                "plan": {
                    "1h": {"entries": [], "targets": []}
                }
            }
        }

    def test_tg_formatter_renders_note_when_flag_on(self):
        os.environ["TB_DIGEST_EXPLAIN_HIGH_RISK_ACTION"] = "1"
        txt = render_digest(
            timestamp_utc="2025-08-13T00:00:00Z",
            weekly={},
            engine={"thesis_text": "test"},
            assets_ordered=["BTC/USD"],
            assets_data=self._sample_assets(),
            options={"include_weekly": False, "include_engine": False},
            polymarket=[],
        )
        self.assertIn("⚠ Extreme-risk long setup", txt)

    def test_tg_formatter_omits_note_when_flag_off(self):
        os.environ["TB_DIGEST_EXPLAIN_HIGH_RISK_ACTION"] = "0"
        txt = render_digest(
            timestamp_utc="2025-08-13T00:00:00Z",
            weekly={},
            engine={"thesis_text": "test"},
            assets_ordered=["BTC/USD"],
            assets_data=self._sample_assets(),
            options={"include_weekly": False, "include_engine": False},
            polymarket=[],
        )
        self.assertNotIn("⚠ Extreme-risk long setup", txt)

    def test_discord_formatter_renders_note_when_flag_on(self):
        os.environ["TB_DIGEST_EXPLAIN_HIGH_RISK_ACTION"] = "1"
        digest_data = {
            "timestamp": "2025-08-13T00:00:00Z",
            "executive_take": "test",
            "weekly": {},
            "engine": {"thesis_text": "test"},
            "assets": [
                {
                    "symbol": "BTC/USD",
                    "risk": "high",
                    "readiness": "Now",
                    "action": "Buy",
                    "risk_score": 0.9,
                    "plan": {"1h": {"entries": [], "targets": []}},
                }
            ],
            "polymarket": [],
        }
        embeds = digest_to_discord_embeds(digest_data)
        all_fields = []
        for e in embeds:
            all_fields.extend(e.get("fields", []))
        # Find Evidence field content
        ev_vals = [f["value"] for f in all_fields if f.get("name") == "Evidence"]
        self.assertTrue(any("Extreme-risk long setup" in v for v in ev_vals))

    def test_discord_formatter_omits_note_when_flag_off(self):
        os.environ["TB_DIGEST_EXPLAIN_HIGH_RISK_ACTION"] = "0"
        digest_data = {
            "timestamp": "2025-08-13T00:00:00Z",
            "executive_take": "test",
            "weekly": {},
            "engine": {"thesis_text": "test"},
            "assets": [
                {
                    "symbol": "BTC/USD",
                    "risk": "high",
                    "readiness": "Now",
                    "action": "Buy",
                    "risk_score": 0.9,
                    "plan": {"1h": {"entries": [], "targets": []}},
                }
            ],
            "polymarket": [],
        }
        embeds = digest_to_discord_embeds(digest_data)
        all_fields = []
        for e in embeds:
            all_fields.extend(e.get("fields", []))
        ev_vals = [f["value"] for f in all_fields if f.get("name") == "Evidence"]
        self.assertFalse(any("Extreme-risk long setup" in v for v in ev_vals))


if __name__ == "__main__":
    unittest.main()
