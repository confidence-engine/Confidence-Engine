"""
Trading hours utilities for multi-asset support.
"""

from datetime import datetime, timezone
from typing import Dict, Optional

from symbol_utils import is_crypto, is_stock


def trading_hours_state(symbol: str, when_utc: Optional[datetime] = None) -> Dict[str, str]:
    """
    Get trading hours state for a symbol.
    
    Args:
        symbol: Symbol string
        when_utc: UTC timestamp (defaults to now)
        
    Returns:
        Dictionary with "state" and "tz" keys
    """
    if when_utc is None:
        when_utc = datetime.now(timezone.utc)
    
    # Ensure we have a timezone-aware datetime
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=timezone.utc)
    
    if is_crypto(symbol):
        return {
            "state": "24x7",
            "tz": "24x7"
        }
    
    elif is_stock(symbol):
        # US market hours: 9:30 AM - 4:00 PM ET
        # Convert to UTC: ET is UTC-5 (EST) or UTC-4 (EDT)
        # For simplicity, we'll use UTC-5 (EST) year-round
        # This means US market hours are 14:30-21:00 UTC
        
        hour_utc = when_utc.hour
        
        if 14 <= hour_utc < 21:  # 14:30-21:00 UTC (inclusive start, exclusive end)
            # For 14:00-14:29, check if we're past 14:30
            if hour_utc == 14 and when_utc.minute < 30:
                return {
                    "state": "CLOSED",
                    "tz": "America/New_York"
                }
            return {
                "state": "RTH",  # Regular Trading Hours
                "tz": "America/New_York"
            }
        else:  # Outside RTH
            return {
                "state": "CLOSED",
                "tz": "America/New_York"
            }
    
    else:
        # Unknown symbol type
        return {
            "state": "UNKNOWN",
            "tz": "UTC"
        }


def is_market_open(symbol: str, when_utc: Optional[datetime] = None) -> bool:
    """
    Check if market is open for a symbol.
    
    Args:
        symbol: Symbol string
        when_utc: UTC timestamp (defaults to now)
        
    Returns:
        True if market is open, False otherwise
    """
    state_info = trading_hours_state(symbol, when_utc)
    return state_info["state"] in ["RTH", "EXT", "24x7"]


def get_market_hours_info(symbol: str, when_utc: Optional[datetime] = None) -> Dict[str, str]:
    """
    Get comprehensive market hours information.
    
    Args:
        symbol: Symbol string
        when_utc: UTC timestamp (defaults to now)
        
    Returns:
        Dictionary with market hours information
    """
    state_info = trading_hours_state(symbol, when_utc)
    
    if is_crypto(symbol):
        return {
            **state_info,
            "description": "24/7 trading",
            "next_open": None,
            "next_close": None
        }
    
    elif is_stock(symbol):
        # For stocks, we could add more detailed information
        # about next open/close times, but for now keep it simple
        return {
            **state_info,
            "description": "US market hours (9:30 AM - 4:00 PM ET)",
            "next_open": None,  # Could be calculated
            "next_close": None  # Could be calculated
        }
    
    else:
        return {
            **state_info,
            "description": "Unknown market hours",
            "next_open": None,
            "next_close": None
        }
