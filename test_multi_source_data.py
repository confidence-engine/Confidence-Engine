#!/usr/bin/env python3
"""
Test script for multi-source data provider
Tests all available data sources and compares with Alpaca
"""

import os
import sys
import pandas as pd
from datetime import datetime, timezone

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_source_data_provider import MultiSourceDataProvider

def test_data_providers():
    """Test all data providers and compare results"""

    print("🧪 Testing Multi-Source Data Provider")
    print("=" * 60)

    provider = MultiSourceDataProvider()

    # Check provider status
    print("📊 Provider Status:")
    status = provider.get_provider_status()
    for name, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {name}: {'Available' if available else 'Not Available'}")
    print()

    # Test symbols
    test_symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    timeframes = ["15Min", "1Hour"]

    for symbol in test_symbols:
        print(f"🪙 Testing {symbol}")
        print("-" * 40)

        for timeframe in timeframes:
            print(f"  ⏰ {timeframe} timeframe:")

            try:
                # Get data from multi-source provider
                data = provider.get_crypto_bars(symbol, timeframe, limit=5)

                if data is not None and not data.empty:
                    print(f"    ✅ Got {len(data)} bars")
                    print(f"    📅 Latest: {data.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    print(".2f")
                    print(".2f")
                    print(".0f")
                else:
                    print("    ❌ No data received")
            except Exception as e:
                print(f"    ❌ Error: {e}")

            print()

        # Test news (limited availability)
        try:
            news = provider.get_crypto_news(symbol, limit=3)
            if news:
                print(f"  📰 News headlines: {len(news)} found")
                for i, headline in enumerate(news[:2], 1):
                    print(f"    {i}. {headline[:60]}...")
            else:
                print("  📰 No news headlines available")
        except Exception as e:
            print(f"  📰 News error: {e}")

        print()

def compare_with_alpaca():
    """Compare multi-source data with Alpaca (if available)"""

    print("🔄 Comparing with Alpaca (if available)")
    print("=" * 60)

    try:
        from alpaca import recent_bars
        alpaca_available = True
    except ImportError:
        alpaca_available = False
        print("❌ Alpaca not available for comparison")
        return

    if not alpaca_available:
        return

    provider = MultiSourceDataProvider()
    symbol = "BTC/USD"
    timeframe = "15Min"

    print(f"📊 Comparing {symbol} {timeframe} data:")
    print()

    # Get Alpaca data
    try:
        alpaca_data = recent_bars(symbol, minutes=60)  # Last hour
        if alpaca_data is not None and not alpaca_data.empty:
            alpaca_latest = alpaca_data.index[-1]
            alpaca_price = alpaca_data['close'].iloc[-1]
            print("  🏦 Alpaca:")
            print(f"    📅 Latest: {alpaca_latest.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(".2f")
    except Exception as e:
        print(f"  🏦 Alpaca error: {e}")

    # Get multi-source data
    try:
        multi_data = provider.get_crypto_bars(symbol, timeframe, limit=10)
        if multi_data is not None and not multi_data.empty:
            multi_latest = multi_data.index[-1]
            multi_price = multi_data['close'].iloc[-1]
            print("  🔄 Multi-Source:")
            print(f"    📅 Latest: {multi_latest.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(".2f")
    except Exception as e:
        print(f"  🔄 Multi-Source error: {e}")

def show_usage_examples():
    """Show how to use the multi-source provider in your code"""

    print("📚 Usage Examples")
    print("=" * 60)

    usage_code = '''
# Basic usage in your trading code:

from multi_source_data_provider import get_crypto_bars_alternative

# Replace Alpaca calls with multi-source provider
def fetch_bars(symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
    """Enhanced fetch_bars with multiple free data sources"""
    data = get_crypto_bars_alternative(symbol, timeframe, lookback)
    if data is None:
        # Fallback to Alpaca if needed
        from alpaca import recent_bars
        return recent_bars(symbol, minutes=lookback)
    return data

# Configuration in .env:
# TB_ENABLE_MULTI_SOURCE_DATA=1
# TB_PREFERRED_DATA_SOURCE=yahoo  # yahoo, binance, coingecko, alphavantage
# TB_DATA_SOURCE_FAILOVER=1       # Auto failover if one source fails
'''

    print(usage_code)

if __name__ == "__main__":
    test_data_providers()
    compare_with_alpaca()
    show_usage_examples()
