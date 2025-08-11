"""
Symbol utilities for multi-asset support.
"""

import re
from typing import Dict, List, Optional


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol to canonical form.
    
    Args:
        symbol: Raw symbol string
        
    Returns:
        Canonical symbol string
    """
    if not symbol:
        return ""
    
    # Remove whitespace and convert to uppercase
    normalized = symbol.strip().upper()
    
    # Handle crypto pairs - ensure /USD format
    # Convert BTCUSD -> BTC/USD, ETHUSD -> ETH/USD
    if '/' not in normalized and len(normalized) >= 6 and normalized.endswith('USD') and '-' not in normalized and '_' not in normalized:
        # Assume last 3 chars are USD
        base = normalized[:-3]
        quote = normalized[-3:]
        if quote == 'USD':
            normalized = f"{base}/{quote}"
    
    return normalized


def is_crypto(symbol: str) -> bool:
    """
    Check if symbol is a cryptocurrency.
    
    Args:
        symbol: Symbol string
        
    Returns:
        True if crypto, False otherwise
    """
    if not symbol:
        return False
    
    normalized = normalize_symbol(symbol)
    
    # Common crypto patterns
    crypto_patterns = [
        r'^BTC(/USD)?$',
        r'^ETH(/USD)?$',
        r'^ADA(/USD)?$',
        r'^DOT(/USD)?$',
        r'^LINK(/USD)?$',
        r'^LTC(/USD)?$',
        r'^XRP(/USD)?$',
        r'^SOL(/USD)?$',
        r'^MATIC(/USD)?$',
        r'^AVAX(/USD)?$',
        r'^UNI(/USD)?$',
        r'^ATOM(/USD)?$',
        r'^NEAR(/USD)?$',
        r'^FTM(/USD)?$',
        r'^ALGO(/USD)?$',
        r'^VET(/USD)?$',
        r'^ICP(/USD)?$',
        r'^FIL(/USD)?$',
        r'^THETA(/USD)?$',
        r'^XTZ(/USD)?$',
        r'^HBAR(/USD)?$',
        r'^EGLD(/USD)?$',
        r'^MANA(/USD)?$',
        r'^SAND(/USD)?$',
        r'^AXS(/USD)?$',
        r'^GALA(/USD)?$',
        r'^ENJ(/USD)?$',
        r'^CHZ(/USD)?$',
        r'^HOT(/USD)?$',
        r'^BAT(/USD)?$',
        r'^ZIL(/USD)?$',
        r'^ONE(/USD)?$',
        r'^IOTA(/USD)?$',
        r'^NEO(/USD)?$',
        r'^VET(/USD)?$',
        r'^TRX(/USD)?$',
        r'^EOS(/USD)?$',
        r'^XLM(/USD)?$',
        r'^XMR(/USD)?$',
        # r'^DASH(/USD)?$',  # Commented out as DASH is also a stock symbol
        r'^ZEC(/USD)?$',
        r'^BCH(/USD)?$',
        r'^BSV(/USD)?$',
        r'^ETC(/USD)?$',
        r'^DOGE(/USD)?$',
        r'^SHIB(/USD)?$',
        r'^PEPE(/USD)?$',
        r'^FLOKI(/USD)?$',
        r'^BONK(/USD)?$',
        r'^WIF(/USD)?$',
        r'^JUP(/USD)?$',
        r'^PYTH(/USD)?$',
        r'^JTO(/USD)?$',
        r'^WEN(/USD)?$',
        r'^MYRO(/USD)?$',
        r'^POPCAT(/USD)?$',
        r'^BOOK(/USD)?$',
        r'^TURBO(/USD)?$',
        r'^SMOG(/USD)?$',
        r'^SLERF(/USD)?$',
        r'^SLOTH(/USD)?$',
        r'^MEW(/USD)?$',
        r'^CAT(/USD)?$',
        r'^DOG(/USD)?$',
        r'^MOON(/USD)?$',
        r'^ROCKET(/USD)?$',
        r'^LAMBO(/USD)?$',
        r'^HODL(/USD)?$',
        r'^APE(/USD)?$',
        r'^DEGEN(/USD)?$',
        r'^FOMO(/USD)?$',
        r'^PUMP(/USD)?$',
        r'^DUMP(/USD)?$',
        r'^BULL(/USD)?$',
        r'^BEAR(/USD)?$',
        r'^DIAMOND(/USD)?$',
        r'^HANDS(/USD)?$',
        r'^TENDIES(/USD)?$',
        r'^STONKS(/USD)?$',
        r'^YOLO(/USD)?$',
        r'^WAGMI(/USD)?$',
        r'^NGMI(/USD)?$',
        r'^FUD(/USD)?$',
        r'^FOMO(/USD)?$',
        r'^HODL(/USD)?$',
        r'^LAMBO(/USD)?$',
        r'^MOON(/USD)?$',
        r'^ROCKET(/USD)?$',
        r'^STONKS(/USD)?$',
        r'^TENDIES(/USD)?$',
        r'^WAGMI(/USD)?$',
        r'^YOLO(/USD)?$',
    ]
    
    for pattern in crypto_patterns:
        if re.match(pattern, normalized):
            return True
    
    return False


def is_stock(symbol: str) -> bool:
    """
    Check if symbol is a stock.
    
    Args:
        symbol: Symbol string
        
    Returns:
        True if stock, False otherwise
    """
    if not symbol:
        return False
    
    normalized = normalize_symbol(symbol)
    
    # Check if it's a known crypto first
    if is_crypto(normalized):
        return False
    
    # Check if it looks like a valid stock symbol
    # Stock symbols are typically 1-5 characters, uppercase letters
    if len(normalized) <= 5 and normalized.isalpha():
        return True
    
    # For now, be conservative - only return True for known patterns
    return False


def get_symbol_type(symbol: str) -> str:
    """
    Get the type of a symbol.
    
    Args:
        symbol: Symbol string
        
    Returns:
        "crypto", "stock", or "unknown"
    """
    if is_crypto(symbol):
        return "crypto"
    elif is_stock(symbol):
        return "stock"
    else:
        return "unknown"


def validate_universe_config(universe: Dict[str, List[str]]) -> bool:
    """
    Validate universe configuration.
    
    Args:
        universe: Dictionary with "crypto" and "stocks" lists
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(universe, dict):
        return False
    
    if "crypto" not in universe or "stocks" not in universe:
        return False
    
    if not isinstance(universe["crypto"], list) or not isinstance(universe["stocks"], list):
        return False
    
    # Validate each symbol
    for symbol in universe["crypto"]:
        if not is_crypto(symbol):
            return False
    
    for symbol in universe["stocks"]:
        if not is_stock(symbol):
            return False
    
    return True
