"""
Stock bars adapter for multi-asset support.
"""

import os
import time
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def get_bars_stock(symbol: str, lookback_minutes: int) -> List[Dict]:
    """
    Get stock bars data.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "SPY")
        lookback_minutes: Number of minutes to look back
        
    Returns:
        List of bar dictionaries with keys: ts, close, volume
    """
    # Check if we should use stub data
    if os.getenv("TB_ALLOW_STUB_BARS") == "1":
        logger.info(f"Using stub bars for {symbol}")
        return _get_stub_bars_stock(symbol, lookback_minutes)
    
    # Try to get real data
    try:
        return _get_real_bars_stock(symbol, lookback_minutes)
    except Exception as e:
        logger.warning(f"Failed to get real bars for {symbol}: {e}")
        logger.info("Falling back to stub bars")
        return _get_stub_bars_stock(symbol, lookback_minutes)


def _get_real_bars_stock(symbol: str, lookback_minutes: int) -> List[Dict]:
    """
    Get real stock bars data.
    
    This is a placeholder implementation. In production, you would:
    1. Use a real stock data provider (Yahoo Finance, Alpha Vantage, etc.)
    2. Implement proper rate limiting and error handling
    3. Add authentication and API key management
    
    Args:
        symbol: Stock symbol
        lookback_minutes: Number of minutes to look back
        
    Returns:
        List of bar dictionaries
    """
    # For now, raise an exception to trigger fallback to stub data
    raise NotImplementedError("Real stock data provider not implemented")


def _get_stub_bars_stock(symbol: str, lookback_minutes: int) -> List[Dict]:
    """
    Get stub stock bars data for testing.
    
    Args:
        symbol: Stock symbol
        lookback_minutes: Number of minutes to look back
        
    Returns:
        List of bar dictionaries
    """
    import random
    from datetime import datetime, timezone, timedelta
    
    # Generate deterministic but realistic-looking data
    base_price = 100.0
    if symbol == "AAPL":
        base_price = 150.0
    elif symbol == "MSFT":
        base_price = 300.0
    elif symbol == "SPY":
        base_price = 400.0
    
    bars = []
    now = datetime.now(timezone.utc)
    
    # Generate bars going backwards in time
    for i in range(max(1, lookback_minutes)):  # Ensure at least 1 bar
        # Create timestamp (1-minute intervals)
        timestamp = now - timedelta(minutes=i)
        ts = int(timestamp.timestamp())
        
        # Generate price with some randomness
        # Use symbol as seed for deterministic but different data per symbol
        random.seed(hash(symbol) + i)
        
        # Add some trend and noise
        trend = (i / max(1, lookback_minutes)) * 0.02  # Small trend
        noise = (random.random() - 0.5) * 0.01  # Small noise
        price_change = trend + noise
        
        close = base_price * (1 + price_change)
        
        # Generate volume (higher during market hours)
        hour = timestamp.hour
        if 14 <= hour < 21:  # Market hours (UTC)
            base_volume = 1000000
        else:
            base_volume = 100000
        
        volume = base_volume * (0.5 + random.random())
        
        bars.append({
            "ts": ts,
            "close": round(close, 2),
            "volume": int(volume)
        })
    
    # Reverse to get chronological order (oldest first)
    bars.reverse()
    
    return bars


def _get_yahoo_bars_stock(symbol: str, lookback_minutes: int) -> List[Dict]:
    """
    Get stock bars from Yahoo Finance (placeholder).
    
    This would be implemented with yfinance or similar library.
    
    Args:
        symbol: Stock symbol
        lookback_minutes: Number of minutes to look back
        
    Returns:
        List of bar dictionaries
    """
    # Placeholder for Yahoo Finance implementation
    # import yfinance as yf
    # 
    # ticker = yf.Ticker(symbol)
    # end_time = datetime.now()
    # start_time = end_time - timedelta(minutes=lookback_minutes)
    # 
    # data = ticker.history(start=start_time, end=end_time, interval="1m")
    # 
    # bars = []
    # for timestamp, row in data.iterrows():
    #     bars.append({
    #         "ts": int(timestamp.timestamp()),
    #         "close": float(row["Close"]),
    #         "volume": float(row["Volume"])
    #     })
    # 
    # return bars
    
    raise NotImplementedError("Yahoo Finance integration not implemented")
