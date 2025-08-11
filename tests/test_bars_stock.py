"""
Tests for stock bars adapter.
"""

import pytest
import os
from unittest.mock import patch

from bars_stock import get_bars_stock


def test_get_bars_stock_stub_data():
    """Test stock bars with stub data."""
    # Set environment to use stub data
    with patch.dict(os.environ, {"TB_ALLOW_STUB_BARS": "1"}):
        bars = get_bars_stock("AAPL", 60)
        
        # Should return list of bars
        assert isinstance(bars, list)
        assert len(bars) >= 20  # Should have at least 20 bars
        
        # Check bar structure
        for bar in bars:
            assert "ts" in bar
            assert "close" in bar
            assert "volume" in bar
            
            # Check data types
            assert isinstance(bar["ts"], int)
            assert isinstance(bar["close"], float)
            assert isinstance(bar["volume"], int)
            
            # Check reasonable values
            assert bar["close"] > 0
            assert bar["volume"] > 0


def test_get_bars_stock_different_symbols():
    """Test stock bars for different symbols."""
    with patch.dict(os.environ, {"TB_ALLOW_STUB_BARS": "1"}):
        symbols = ["AAPL", "MSFT", "SPY"]
        
        for symbol in symbols:
            bars = get_bars_stock(symbol, 30)
            
            assert len(bars) >= 20
            
            # Check that different symbols have different base prices
            avg_price = sum(bar["close"] for bar in bars) / len(bars)
            
            if symbol == "AAPL":
                assert 100 < avg_price < 200  # Around 150
            elif symbol == "MSFT":
                assert 200 < avg_price < 400  # Around 300
            elif symbol == "SPY":
                assert 300 < avg_price < 500  # Around 400


def test_get_bars_stock_timestamp_ordering():
    """Test that bars are ordered by timestamp."""
    with patch.dict(os.environ, {"TB_ALLOW_STUB_BARS": "1"}):
        bars = get_bars_stock("AAPL", 60)
        
        # Check that timestamps are in ascending order
        timestamps = [bar["ts"] for bar in bars]
        assert timestamps == sorted(timestamps)
        
        # Check that timestamps are reasonable (within last hour)
        import time
        current_time = int(time.time())
        
        for ts in timestamps:
            assert current_time - 3600 <= ts <= current_time  # Within last hour


def test_get_bars_stock_lookback_variation():
    """Test different lookback periods."""
    with patch.dict(os.environ, {"TB_ALLOW_STUB_BARS": "1"}):
        lookbacks = [10, 30, 60, 120]
        
        for lookback in lookbacks:
            bars = get_bars_stock("AAPL", lookback)
            
            # Should have approximately the requested number of bars
            assert len(bars) >= min(lookback, 20)  # At least requested or 20, whichever is smaller
            
            # Check that bars span the requested time period
            if len(bars) >= 2:
                time_span = bars[-1]["ts"] - bars[0]["ts"]
                expected_span = (lookback - 1) * 60  # Convert minutes to seconds
                
                # Allow some tolerance for rounding
                assert abs(time_span - expected_span) <= 120  # Within 2 minutes


def test_get_bars_stock_deterministic():
    """Test that stub data is deterministic for same symbol and lookback."""
    with patch.dict(os.environ, {"TB_ALLOW_STUB_BARS": "1"}):
        # Get bars twice for same symbol and lookback
        bars1 = get_bars_stock("AAPL", 30)
        bars2 = get_bars_stock("AAPL", 30)
        
        # Should be identical
        assert len(bars1) == len(bars2)
        
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            assert bar1["ts"] == bar2["ts"], f"Timestamp mismatch at index {i}"
            assert bar1["close"] == bar2["close"], f"Close mismatch at index {i}"
            assert bar1["volume"] == bar2["volume"], f"Volume mismatch at index {i}"


def test_get_bars_stock_different_symbols_different_data():
    """Test that different symbols produce different data."""
    with patch.dict(os.environ, {"TB_ALLOW_STUB_BARS": "1"}):
        bars_aapl = get_bars_stock("AAPL", 30)
        bars_msft = get_bars_stock("MSFT", 30)
        
        # Should have same number of bars
        assert len(bars_aapl) == len(bars_msft)
        
        # But different data
        aapl_closes = [bar["close"] for bar in bars_aapl]
        msft_closes = [bar["close"] for bar in bars_msft]
        
        assert aapl_closes != msft_closes  # Different close prices
        
        # Different average prices
        avg_aapl = sum(aapl_closes) / len(aapl_closes)
        avg_msft = sum(msft_closes) / len(msft_closes)
        
        assert abs(avg_aapl - avg_msft) > 50  # Significantly different


def test_get_bars_stock_fallback_to_stub():
    """Test fallback to stub data when real data fails."""
    # Don't set TB_ALLOW_STUB_BARS, but mock the real data function to fail
    with patch('bars_stock._get_real_bars_stock', side_effect=Exception("API error")):
        bars = get_bars_stock("AAPL", 30)
        
        # Should still return stub data
        assert isinstance(bars, list)
        assert len(bars) >= 20
        
        # Check structure
        for bar in bars:
            assert "ts" in bar
            assert "close" in bar
            assert "volume" in bar


def test_get_bars_stock_edge_cases():
    """Test edge cases for stock bars."""
    with patch.dict(os.environ, {"TB_ALLOW_STUB_BARS": "1"}):
        # Test very small lookback
        bars = get_bars_stock("AAPL", 1)
        assert len(bars) >= 1
        
        # Test zero lookback (should handle gracefully)
        bars = get_bars_stock("AAPL", 0)
        assert len(bars) >= 1  # Should still return some data
        
        # Test very large lookback
        bars = get_bars_stock("AAPL", 1000)
        assert len(bars) >= 20  # Should have reasonable amount of data


def test_get_bars_stock_data_quality():
    """Test quality of generated data."""
    with patch.dict(os.environ, {"TB_ALLOW_STUB_BARS": "1"}):
        bars = get_bars_stock("AAPL", 60)
        
        # Check for reasonable price movements
        closes = [bar["close"] for bar in bars]
        
        # Should have some variation
        price_range = max(closes) - min(closes)
        assert price_range > 0
        
        # Should not have extreme movements (within reasonable bounds)
        for i in range(1, len(closes)):
            change_pct = abs(closes[i] - closes[i-1]) / closes[i-1]
            assert change_pct < 0.1  # No more than 10% change between bars
        
        # Check volume data
        volumes = [bar["volume"] for bar in bars]
        assert all(v > 0 for v in volumes)
        
        # Should have some variation in volume
        volume_range = max(volumes) - min(volumes)
        assert volume_range > 0
