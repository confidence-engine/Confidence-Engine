"""
Tests for universe configuration loader.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from symbol_utils import validate_universe_config


def test_validate_universe_config_valid():
    """Test valid universe configuration."""
    config = {
        "crypto": ["BTC/USD", "ETH/USD"],
        "stocks": ["AAPL", "MSFT", "SPY"]
    }
    
    assert validate_universe_config(config) is True


def test_validate_universe_config_invalid_structure():
    """Test invalid universe configuration structure."""
    # Missing required keys
    config1 = {"crypto": ["BTC/USD"]}
    assert validate_universe_config(config1) is False
    
    # Wrong types
    config2 = {"crypto": "BTC/USD", "stocks": ["AAPL"]}
    assert validate_universe_config(config2) is False
    
    # Not a dict
    config3 = ["BTC/USD", "AAPL"]
    assert validate_universe_config(config3) is False


def test_validate_universe_config_invalid_symbols():
    """Test universe config with invalid symbols."""
    config = {
        "crypto": ["INVALID", "ETH/USD"],
        "stocks": ["AAPL", "INVALID"]
    }
    
    assert validate_universe_config(config) is False


def test_load_universe_config_from_file():
    """Test loading universe config from YAML file."""
    # Create a temporary YAML file
    config_data = {
        "crypto": ["BTC/USD", "ETH/USD"],
        "stocks": ["AAPL", "MSFT", "SPY"]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        # Test loading
        with open(temp_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config == config_data
        assert validate_universe_config(loaded_config) is True
        
    finally:
        # Clean up
        Path(temp_path).unlink()


def test_universe_config_symbol_types():
    """Test that universe config contains correct symbol types."""
    config = {
        "crypto": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "stocks": ["AAPL", "MSFT", "SPY", "TSLA"]
    }
    
    assert validate_universe_config(config) is True
    
    # Verify crypto symbols are recognized as crypto
    from symbol_utils import is_crypto, is_stock
    
    for symbol in config["crypto"]:
        assert is_crypto(symbol), f"{symbol} should be recognized as crypto"
        assert not is_stock(symbol), f"{symbol} should not be recognized as stock"
    
    for symbol in config["stocks"]:
        assert is_stock(symbol), f"{symbol} should be recognized as stock"
        assert not is_crypto(symbol), f"{symbol} should not be recognized as crypto"
