import re
import sys
from pathlib import Path

# Ensure project root on path for direct execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evidence_lines import generate_evidence_line, strip_numbers_for_chat
from scripts.tracer_bullet_universe import select_digest_symbols_public


def test_strip_numbers_basic():
    s = "Price up 12.5% with +3.2R target at 1,234.56"
    out = strip_numbers_for_chat(s)
    assert not re.search(r"\d", out), f"Found digits in output: {out}"


def test_generate_evidence_line_no_numbers():
    line = generate_evidence_line(
        sentiment_tag="bullish",
        participation_tag="hot",
        tf_aligned=True,
        signal_quality_tag="strong",
        narrative_tags=["continuation", "trend"],
    )
    assert isinstance(line, str) and len(line) > 0
    assert not re.search(r"\d", line), f"Evidence line contains digits: {line}"


def test_evidence_line_for_surfaced_assets():
    # Build a dummy ranked payload list with BTC, ETH, and alts
    payloads = [
        {"symbol": "BTC/USD", "symbol_type": "crypto"},
        {"symbol": "ETH/USD", "symbol_type": "crypto"},
        {"symbol": "XRP/USD", "symbol_type": "crypto"},
        {"symbol": "SOL/USD", "symbol_type": "crypto"},
        {"symbol": "OP/USD", "symbol_type": "crypto"},
        {"symbol": "ARB/USD", "symbol_type": "crypto"},
    ]

    surfaced = select_digest_symbols_public(payloads, k_alts=3)
    # Should include BTC, ETH + 3 alts
    assert len(surfaced) == 5

    # For each surfaced asset, ensure we can generate a number-free evidence line
    for sym in surfaced:
        line = generate_evidence_line(
            sentiment_tag="bullish" if "BTC" in sym else "bearish" if "ETH" in sym else "watch",
            participation_tag="quiet",
            tf_aligned=True,
            signal_quality_tag="strong",
            narrative_tags=["trend"],
        )
        assert isinstance(line, str) and len(line) > 0
        assert not re.search(r"\d", line), f"Digits found in line for {sym}: {line}"
