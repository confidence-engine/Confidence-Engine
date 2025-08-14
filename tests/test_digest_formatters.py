import os
import unittest

from scripts.tg_digest_formatter import render_digest
from scripts.discord_formatter import digest_to_discord_embeds

SAMPLE_ASSET = {
    'symbol': 'BTC/USD',
    'thesis': {
        'action': 'buy',
        'readiness': 'near',
        'risk_band': 'Medium',
        'tf_aligned': True,
    },
    'alignment_flag': True,
    'signal_quality': 'strong',
    'structure': 'trend continuation',
    'spot': 64000.0,
    'confirmation_checks': [
        {'name': 'price_vs_narrative', 'passed': True},
        {'name': 'volume_support', 'passed': True},
        {'name': 'timescale_alignment', 'passed': True},
    ],
    'plan': {
        '1h': {
            'source': 'analysis',
            'explain': 'Momentum break with MA confluence',
            'entries': [{'type': 'breakout', 'zone_or_trigger': 65000}],
            'invalidation': {'price': 63000, 'condition': 'breach'},
            'targets': [{'label': 'TP1', 'price': 67000}],
        },
        '4h': {
            'source': 'fallback',
            'entries': [{'type': 'zone', 'zone_or_trigger': [62000, 64000]}],
            'invalidation': {'price': 61000},
            'targets': [{'label': 'TP1', 'price': 66000}],
        },
    },
}

SAMPLE_DIGEST = {
    'timestamp': '2025-08-15 12:00 UTC',
    'executive_take': 'Trade the break or the fail; avoid chasing.',
    'weekly': {'regime': 'mixed'},
    'engine': {'thesis_text': 'Lean into clean triggers.'},
    'assets': [SAMPLE_ASSET],
}


class TestDigestFormatters(unittest.TestCase):
    def test_provenance_mapping_and_tf_grades_telegram(self):
        txt = render_digest(
            timestamp_utc=SAMPLE_DIGEST['timestamp'],
            weekly=SAMPLE_DIGEST['weekly'],
            engine=SAMPLE_DIGEST['engine'],
            assets_ordered=['BTC/USD'],
            assets_data={'BTC/USD': SAMPLE_ASSET},
            options={'include_weekly': False, 'include_engine': False},
        )
        self.assertIn('1h: (agent mode)', txt)
        self.assertIn('4h: (fallback)', txt)
        # per-TF micro grades removed; ensure provenance labels still present
        self.assertNotIn('[Grade:', txt)

    def test_provenance_mapping_and_tf_grades_discord(self):
        embeds = digest_to_discord_embeds(SAMPLE_DIGEST)
        titles = [e.get('title', '') for e in embeds]
        # harmonized header uses thesis fields
        hdr = next(t for t in titles if t.startswith('BTC/USD'))
        self.assertIn('Risk Level: Medium', hdr)
        self.assertIn('Timing: near', hdr)
        self.assertIn('Stance: buy', hdr)
        # per-TF labels must have provenance; grades removed
        tf_fields = []
        for e in embeds:
            for f in e.get('fields', []):
                if f['name'].startswith('1h') or f['name'].startswith('4h'):
                    tf_fields.append(f['name'])
        joined = '\n'.join(tf_fields)
        self.assertIn('1h (agent mode)', joined)
        self.assertIn('4h (fallback)', joined)
        self.assertNotIn('[Grade:', joined)


if __name__ == '__main__':
    unittest.main()
