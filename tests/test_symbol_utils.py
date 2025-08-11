"""
Tests for symbol utilities.
"""

import pytest

from symbol_utils import normalize_symbol, is_crypto, is_stock, get_symbol_type


def test_normalize_symbol():
    """Test symbol normalization."""
    # Test crypto symbols
    assert normalize_symbol("btc/usd") == "BTC/USD"
    assert normalize_symbol("ETH/USD") == "ETH/USD"
    assert normalize_symbol("btcusd") == "BTC/USD"
    assert normalize_symbol("ethusd") == "ETH/USD"
    
    # Test stock symbols
    assert normalize_symbol("aapl") == "AAPL"
    assert normalize_symbol("MSFT") == "MSFT"
    assert normalize_symbol(" spy ") == "SPY"
    
    # Test edge cases
    assert normalize_symbol("") == ""
    assert normalize_symbol("   ") == ""


def test_is_crypto():
    """Test crypto symbol detection."""
    # Valid crypto symbols
    assert is_crypto("BTC/USD") is True
    assert is_crypto("ETH/USD") is True
    assert is_crypto("SOL/USD") is True
    assert is_crypto("btc/usd") is True
    assert is_crypto("btcusd") is True
    
    # Invalid crypto symbols
    assert is_crypto("AAPL") is False
    assert is_crypto("MSFT") is False
    assert is_crypto("SPY") is False
    assert is_crypto("") is False
    assert is_crypto("INVALID") is False


def test_is_stock():
    """Test stock symbol detection."""
    # Valid stock symbols
    assert is_stock("AAPL") is True
    assert is_stock("MSFT") is True
    assert is_stock("SPY") is True
    assert is_stock("aapl") is True
    assert is_stock(" msft ") is True
    
    # Invalid stock symbols
    assert is_stock("BTC/USD") is False
    assert is_stock("ETH/USD") is False
    assert is_stock("") is False


def test_get_symbol_type():
    """Test symbol type detection."""
    # Crypto symbols
    assert get_symbol_type("BTC/USD") == "crypto"
    assert get_symbol_type("ETH/USD") == "crypto"
    assert get_symbol_type("SOL/USD") == "crypto"
    
    # Stock symbols
    assert get_symbol_type("AAPL") == "stock"
    assert get_symbol_type("MSFT") == "stock"
    assert get_symbol_type("SPY") == "stock"
    
    # Unknown symbols
    assert get_symbol_type("INVALID") == "unknown"
    assert get_symbol_type("") == "unknown"


def test_crypto_patterns():
    """Test various crypto symbol patterns."""
    crypto_symbols = [
        "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
        "LINK/USD", "LTC/USD", "XRP/USD", "MATIC/USD", "AVAX/USD",
        "UNI/USD", "ATOM/USD", "NEAR/USD", "FTM/USD", "ALGO/USD",
        "VET/USD", "ICP/USD", "FIL/USD", "THETA/USD", "XTZ/USD",
        "HBAR/USD", "EGLD/USD", "MANA/USD", "SAND/USD", "AXS/USD",
        "GALA/USD", "ENJ/USD", "CHZ/USD", "HOT/USD", "BAT/USD",
        "ZIL/USD", "ONE/USD", "IOTA/USD", "NEO/USD", "TRX/USD",
                    "EOS/USD", "XLM/USD", "XMR/USD", "ZEC/USD",  # DASH removed as it conflicts with stock
        "BCH/USD", "BSV/USD", "ETC/USD", "DOGE/USD", "SHIB/USD",
        "PEPE/USD", "FLOKI/USD", "BONK/USD", "WIF/USD", "JUP/USD",
        "PYTH/USD", "JTO/USD", "WEN/USD", "MYRO/USD", "POPCAT/USD",
        "BOOK/USD", "TURBO/USD", "SMOG/USD", "SLERF/USD", "SLOTH/USD",
        "MEW/USD", "CAT/USD", "DOG/USD", "MOON/USD", "ROCKET/USD",
        "LAMBO/USD", "HODL/USD", "APE/USD", "DEGEN/USD", "FOMO/USD",
        "PUMP/USD", "DUMP/USD", "BULL/USD", "BEAR/USD", "DIAMOND/USD",
        "HANDS/USD", "TENDIES/USD", "STONKS/USD", "YOLO/USD",
        "WAGMI/USD", "NGMI/USD", "FUD/USD"
    ]
    
    for symbol in crypto_symbols:
        assert is_crypto(symbol), f"{symbol} should be recognized as crypto"
        assert get_symbol_type(symbol) == "crypto", f"{symbol} should have type 'crypto'"


def test_stock_patterns():
    """Test various stock symbol patterns."""
    stock_symbols = [
        "AAPL", "MSFT", "SPY", "TSLA", "GOOGL", "AMZN", "META",
        "NVDA", "NFLX", "AMD", "INTC", "CRM", "ADBE", "PYPL",
        "COIN", "SQ", "ZM", "UBER", "LYFT", "DASH", "SNOW",
        "PLTR", "RBLX", "HOOD", "COIN", "MSTR", "RIOT", "MARA"
    ]
    
    for symbol in stock_symbols:
        assert is_stock(symbol), f"{symbol} should be recognized as stock"
        assert get_symbol_type(symbol) == "stock", f"{symbol} should have type 'stock'"


def test_normalization_edge_cases():
    """Test edge cases in symbol normalization."""
    # Test with various separators and formats
    assert normalize_symbol("BTC-USD") == "BTC-USD"  # Not crypto pattern
    assert normalize_symbol("BTC_USD") == "BTC_USD"  # Not crypto pattern
    assert normalize_symbol("BTCUSD") == "BTC/USD"   # Should normalize
    
    # Test with numbers
    assert normalize_symbol("AAPL1") == "AAPL1"     # Valid stock-like
    assert normalize_symbol("BTC1/USD") == "BTC1/USD"  # Not in crypto patterns
    
    # Test with special characters
    assert normalize_symbol("AAPL.") == "AAPL."
    assert normalize_symbol("BTC/USD.") == "BTC/USD."
