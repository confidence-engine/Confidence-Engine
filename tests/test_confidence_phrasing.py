import os
import unittest

from scripts.evidence_lines import generate_evidence_line


class TestConfidencePhrasing(unittest.TestCase):
    def test_very_high_aligned(self):
        s = generate_evidence_line(
            sentiment_tag="bullish",
            participation_tag="normal",
            tf_aligned=True,
            signal_quality_tag="very high",
            narrative_tags=["trend"],
        )
        self.assertIn("very high confidence", s)
        self.assertNotIn("mixed", s.lower())

    def test_very_high_fragmented(self):
        s = generate_evidence_line(
            sentiment_tag="bullish",
            participation_tag="normal",
            tf_aligned=False,
            signal_quality_tag="very high",
            narrative_tags=["trend"],
        )
        self.assertIn("dominant timeframe leads", s)
        self.assertIn("very high confidence", s)


if __name__ == "__main__":
    unittest.main()
