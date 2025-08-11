"""
Tests for orchestrator ranking functionality.
"""

import pytest

from scripts.scan_universe import rank_payloads


def test_rank_payloads_basic():
    """Test basic ranking functionality."""
    payloads = [
        {"symbol": "AAPL", "divergence": 0.5, "confidence": 0.7},
        {"symbol": "MSFT", "divergence": 0.3, "confidence": 0.8},
        {"symbol": "SPY", "divergence": 0.8, "confidence": 0.6},
    ]
    
    ranked = rank_payloads(payloads)
    
    # Should be sorted by absolute divergence (descending)
    assert ranked[0]["symbol"] == "SPY"  # abs(0.8) = 0.8
    assert ranked[1]["symbol"] == "AAPL"  # abs(0.5) = 0.5
    assert ranked[2]["symbol"] == "MSFT"  # abs(0.3) = 0.3


def test_rank_payloads_with_timescale_scores():
    """Test ranking with timescale scores."""
    payloads = [
        {
            "symbol": "AAPL",
            "divergence": 0.3,
            "confidence": 0.7,
            "timescale_scores": {"combined_divergence": 0.6}
        },
        {
            "symbol": "MSFT",
            "divergence": 0.5,
            "confidence": 0.8,
            "timescale_scores": {"combined_divergence": 0.2}
        },
        {
            "symbol": "SPY",
            "divergence": 0.4,
            "confidence": 0.6,
            "timescale_scores": {"combined_divergence": 0.8}
        },
    ]
    
    ranked = rank_payloads(payloads)
    
    # Should use combined_divergence when available
    assert ranked[0]["symbol"] == "SPY"  # abs(0.8) = 0.8
    assert ranked[1]["symbol"] == "AAPL"  # abs(0.6) = 0.6
    assert ranked[2]["symbol"] == "MSFT"  # abs(0.2) = 0.2


def test_rank_payloads_mixed_timescale_scores():
    """Test ranking with mixed timescale scores (some have, some don't)."""
    payloads = [
        {
            "symbol": "AAPL",
            "divergence": 0.3,
            "confidence": 0.7,
            "timescale_scores": {"combined_divergence": 0.6}
        },
        {
            "symbol": "MSFT",
            "divergence": 0.5,
            "confidence": 0.8
            # No timescale_scores
        },
        {
            "symbol": "SPY",
            "divergence": 0.4,
            "confidence": 0.6,
            "timescale_scores": {"combined_divergence": 0.8}
        },
    ]
    
    ranked = rank_payloads(payloads)
    
    # Should use combined_divergence when available, fallback to divergence
    assert ranked[0]["symbol"] == "SPY"  # abs(0.8) = 0.8
    assert ranked[1]["symbol"] == "AAPL"  # abs(0.6) = 0.6
    assert ranked[2]["symbol"] == "MSFT"  # abs(0.5) = 0.5


def test_rank_payloads_tie_breaker_confidence():
    """Test ranking with confidence as tie breaker."""
    payloads = [
        {"symbol": "AAPL", "divergence": 0.5, "confidence": 0.7},
        {"symbol": "MSFT", "divergence": 0.5, "confidence": 0.8},  # Same divergence, higher confidence
        {"symbol": "SPY", "divergence": 0.5, "confidence": 0.6},   # Same divergence, lower confidence
    ]
    
    ranked = rank_payloads(payloads)
    
    # Should be sorted by confidence when divergence is equal
    assert ranked[0]["symbol"] == "MSFT"  # Highest confidence
    assert ranked[1]["symbol"] == "AAPL"  # Medium confidence
    assert ranked[2]["symbol"] == "SPY"   # Lowest confidence


def test_rank_payloads_tie_breaker_symbol():
    """Test ranking with symbol as final tie breaker."""
    payloads = [
        {"symbol": "AAPL", "divergence": 0.5, "confidence": 0.7},
        {"symbol": "MSFT", "divergence": 0.5, "confidence": 0.7},  # Same divergence and confidence
        {"symbol": "SPY", "divergence": 0.5, "confidence": 0.7},   # Same divergence and confidence
    ]
    
    ranked = rank_payloads(payloads)
    
    # Should be sorted alphabetically by symbol when everything else is equal
    # Note: With reverse=True, alphabetical order is preserved
    assert ranked[0]["symbol"] == "SPY"   # Alphabetically last (reverse=True)
    assert ranked[1]["symbol"] == "MSFT"  # Alphabetically middle
    assert ranked[2]["symbol"] == "AAPL"  # Alphabetically first


def test_rank_payloads_negative_divergence():
    """Test ranking with negative divergence values."""
    payloads = [
        {"symbol": "AAPL", "divergence": -0.8, "confidence": 0.7},  # abs(-0.8) = 0.8
        {"symbol": "MSFT", "divergence": 0.3, "confidence": 0.8},   # abs(0.3) = 0.3
        {"symbol": "SPY", "divergence": -0.5, "confidence": 0.6},   # abs(-0.5) = 0.5
    ]
    
    ranked = rank_payloads(payloads)
    
    # Should use absolute values for ranking
    assert ranked[0]["symbol"] == "AAPL"  # abs(-0.8) = 0.8
    assert ranked[1]["symbol"] == "SPY"   # abs(-0.5) = 0.5
    assert ranked[2]["symbol"] == "MSFT"  # abs(0.3) = 0.3


def test_rank_payloads_missing_fields():
    """Test ranking with missing fields."""
    payloads = [
        {"symbol": "AAPL", "divergence": 0.5, "confidence": 0.7},
        {"symbol": "MSFT", "confidence": 0.8},  # Missing divergence
        {"symbol": "SPY", "divergence": 0.3},   # Missing confidence
    ]
    
    ranked = rank_payloads(payloads)
    
    # Should handle missing fields gracefully
    assert len(ranked) == 3
    
    # Missing divergence should be treated as 0
    # Missing confidence should be treated as 0
    assert ranked[0]["symbol"] == "AAPL"  # abs(0.5) = 0.5
    assert ranked[1]["symbol"] == "SPY"   # abs(0.3) = 0.3
    assert ranked[2]["symbol"] == "MSFT"  # abs(0) = 0


def test_rank_payloads_empty_list():
    """Test ranking with empty list."""
    payloads = []
    
    ranked = rank_payloads(payloads)
    
    assert ranked == []


def test_rank_payloads_single_item():
    """Test ranking with single item."""
    payloads = [{"symbol": "AAPL", "divergence": 0.5, "confidence": 0.7}]
    
    ranked = rank_payloads(payloads)
    
    assert len(ranked) == 1
    assert ranked[0]["symbol"] == "AAPL"


def test_rank_payloads_complex_scenario():
    """Test ranking with complex scenario."""
    payloads = [
        {
            "symbol": "AAPL",
            "divergence": 0.3,
            "confidence": 0.8,
            "timescale_scores": {"combined_divergence": 0.6}
        },
        {
            "symbol": "MSFT",
            "divergence": 0.5,
            "confidence": 0.7,
            "timescale_scores": {"combined_divergence": 0.5}
        },
        {
            "symbol": "SPY",
            "divergence": 0.4,
            "confidence": 0.9,
            "timescale_scores": {"combined_divergence": 0.5}
        },
        {
            "symbol": "TSLA",
            "divergence": 0.8,
            "confidence": 0.6
            # No timescale_scores
        },
    ]
    
    ranked = rank_payloads(payloads)
    
    # Expected order:
    # 1. TSLA: abs(0.8) = 0.8 (highest divergence)
    # 2. AAPL: abs(0.6) = 0.6 (highest combined_divergence)
    # 3. SPY: abs(0.5) = 0.5, conf 0.9 (higher confidence than MSFT)
    # 4. MSFT: abs(0.5) = 0.5, conf 0.7 (lower confidence than SPY)
    
    assert ranked[0]["symbol"] == "TSLA"
    assert ranked[1]["symbol"] == "AAPL"
    assert ranked[2]["symbol"] == "SPY"
    assert ranked[3]["symbol"] == "MSFT"


def test_rank_payloads_preserves_original_data():
    """Test that ranking preserves original payload data."""
    original_payloads = [
        {
            "symbol": "AAPL",
            "divergence": 0.5,
            "confidence": 0.7,
            "timescale_scores": {"combined_divergence": 0.6},
            "extra_field": "test_value"
        }
    ]
    
    ranked = rank_payloads(original_payloads)
    
    # Should preserve all original data
    assert len(ranked) == 1
    assert ranked[0]["symbol"] == "AAPL"
    assert ranked[0]["divergence"] == 0.5
    assert ranked[0]["confidence"] == 0.7
    assert ranked[0]["timescale_scores"]["combined_divergence"] == 0.6
    assert ranked[0]["extra_field"] == "test_value"
