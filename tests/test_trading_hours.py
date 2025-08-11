"""
Tests for trading hours utilities.
"""

import pytest
from datetime import datetime, timezone

from trading_hours import trading_hours_state, is_market_open, get_market_hours_info


def test_trading_hours_crypto():
    """Test trading hours for crypto symbols."""
    # Crypto should always be 24x7
    crypto_symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    
    for symbol in crypto_symbols:
        state = trading_hours_state(symbol)
        assert state["state"] == "24x7"
        assert state["tz"] == "24x7"
        
        # Should always be open
        assert is_market_open(symbol) is True
        
        # Market hours info
        info = get_market_hours_info(symbol)
        assert info["state"] == "24x7"
        assert info["description"] == "24/7 trading"


def test_trading_hours_stocks_rth():
    """Test trading hours for stocks during regular trading hours."""
    stock_symbols = ["AAPL", "MSFT", "SPY"]
    
    # Create a time during regular trading hours (14:30-21:00 UTC)
    rth_time = datetime(2024, 1, 15, 15, 30, 0, tzinfo=timezone.utc)  # 3:30 PM UTC
    
    for symbol in stock_symbols:
        state = trading_hours_state(symbol, rth_time)
        assert state["state"] == "RTH"
        assert state["tz"] == "America/New_York"
        
        # Should be open
        assert is_market_open(symbol, rth_time) is True
        
        # Market hours info
        info = get_market_hours_info(symbol, rth_time)
        assert info["state"] == "RTH"
        assert "US market hours" in info["description"]


def test_trading_hours_stocks_closed():
    """Test trading hours for stocks when market is closed."""
    stock_symbols = ["AAPL", "MSFT", "SPY"]
    
    # Create a time when market is closed (outside 14:30-21:00 UTC)
    closed_time = datetime(2024, 1, 15, 23, 0, 0, tzinfo=timezone.utc)  # 11:00 PM UTC
    
    for symbol in stock_symbols:
        state = trading_hours_state(symbol, closed_time)
        assert state["state"] == "CLOSED"
        assert state["tz"] == "America/New_York"
        
        # Should be closed
        assert is_market_open(symbol, closed_time) is False
        
        # Market hours info
        info = get_market_hours_info(symbol, closed_time)
        assert info["state"] == "CLOSED"


def test_trading_hours_stocks_extended():
    """Test trading hours for stocks during extended hours."""
    stock_symbols = ["AAPL", "MSFT", "SPY"]
    
    # Create a time during extended hours (outside RTH but not clearly closed)
    # This is a bit tricky since our current logic only has RTH and CLOSED
    # For now, we'll test the boundary conditions
    
    # Test early morning (before RTH)
    early_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)  # 1:00 PM UTC
    
    for symbol in stock_symbols:
        state = trading_hours_state(symbol, early_time)
        assert state["state"] == "CLOSED"  # Current logic treats this as closed
        assert is_market_open(symbol, early_time) is False


def test_trading_hours_edge_cases():
    """Test edge cases for trading hours."""
    # Test with None time (should use current time)
    state = trading_hours_state("BTC/USD")
    assert state["state"] == "24x7"
    
    # Test with timezone-naive datetime
    naive_time = datetime(2024, 1, 15, 15, 30, 0)  # No timezone
    state = trading_hours_state("AAPL", naive_time)
    assert state["state"] in ["RTH", "CLOSED"]  # Should work with auto-timezone
    
    # Test unknown symbol
    state = trading_hours_state("UNKNOWN")
    assert state["state"] == "UNKNOWN"
    assert state["tz"] == "UTC"


def test_market_hours_info():
    """Test comprehensive market hours information."""
    # Test crypto
    crypto_info = get_market_hours_info("BTC/USD")
    assert crypto_info["state"] == "24x7"
    assert crypto_info["description"] == "24/7 trading"
    assert crypto_info["next_open"] is None
    assert crypto_info["next_close"] is None
    
    # Test stock
    stock_time = datetime(2024, 1, 15, 15, 30, 0, tzinfo=timezone.utc)
    stock_info = get_market_hours_info("AAPL", stock_time)
    assert stock_info["state"] == "RTH"
    assert "US market hours" in stock_info["description"]
    assert stock_info["next_open"] is None  # Not implemented yet
    assert stock_info["next_close"] is None  # Not implemented yet


def test_trading_hours_boundaries():
    """Test trading hours boundary conditions."""
    stock_symbol = "AAPL"
    
    # Test RTH start boundary (14:30 UTC)
    rth_start = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    state = trading_hours_state(stock_symbol, rth_start)
    assert state["state"] == "RTH"
    
    # Test RTH end boundary (21:00 UTC)
    rth_end = datetime(2024, 1, 15, 20, 59, 59, tzinfo=timezone.utc)
    state = trading_hours_state(stock_symbol, rth_end)
    assert state["state"] == "RTH"
    
    # Test just after RTH end (21:00 UTC)
    after_rth = datetime(2024, 1, 15, 21, 0, 0, tzinfo=timezone.utc)
    state = trading_hours_state(stock_symbol, after_rth)
    assert state["state"] == "CLOSED"
    
    # Test just before RTH start (14:29 UTC)
    before_rth = datetime(2024, 1, 15, 14, 29, 59, tzinfo=timezone.utc)
    state = trading_hours_state(stock_symbol, before_rth)
    assert state["state"] == "CLOSED"


def test_market_open_logic():
    """Test market open logic for different states."""
    # Crypto should always be open
    assert is_market_open("BTC/USD") is True
    
    # Stock during RTH should be open
    rth_time = datetime(2024, 1, 15, 15, 30, 0, tzinfo=timezone.utc)
    assert is_market_open("AAPL", rth_time) is True
    
    # Stock during closed hours should be closed
    closed_time = datetime(2024, 1, 15, 23, 0, 0, tzinfo=timezone.utc)
    assert is_market_open("AAPL", closed_time) is False
    
    # Unknown symbol should be closed
    assert is_market_open("UNKNOWN") is False
