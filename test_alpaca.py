#!/usr/bin/env python3
"""
Test Alpaca API Integration
Tests basic Alpaca API functionality for paper trading
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def test_alpaca_connection():
    """Test basic Alpaca API connection"""
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass
        
        # Initialize Alpaca client
        api_key = os.getenv("ALPACA_API_KEY_ID")
        secret_key = os.getenv("ALPACA_API_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        
        if not api_key or not secret_key:
            print("❌ Alpaca API credentials not found in environment")
            return False
            
        client = TradingClient(api_key, secret_key, paper=True)
        
        # Test account access
        account = client.get_account()
        print(f"✅ Alpaca connection successful")
        print(f"   Account: {account.account_number}")
        print(f"   Equity: ${float(account.equity):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        
        # Test asset retrieval
        search_params = GetAssetsRequest(asset_class=AssetClass.CRYPTO)
        assets = client.get_all_assets(search_params)
        crypto_assets = [asset for asset in assets if asset.tradable][:5]
        
        print(f"   Available crypto assets: {len(crypto_assets)} (showing first 5)")
        for asset in crypto_assets:
            print(f"     - {asset.symbol}: {asset.name}")
            
        return True
        
    except Exception as e:
        print(f"❌ Alpaca connection failed: {e}")
        return False

def test_market_data():
    """Test market data retrieval"""
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import datetime, timedelta
        
        client = CryptoHistoricalDataClient()
        
        # Test crypto bars
        request_params = CryptoBarsRequest(
            symbol_or_symbols=["BTC/USD"],
            timeframe=TimeFrame.Minute,
            start=datetime.now() - timedelta(hours=1),
            end=datetime.now()
        )
        
        bars = client.get_crypto_bars(request_params)
        
        if bars and len(bars) > 0:
            latest_bar = list(bars["BTC/USD"])[-1]
            print(f"✅ Market data retrieval successful")
            print(f"   Latest BTC/USD: ${latest_bar.close:.2f}")
            print(f"   Volume: {latest_bar.volume:.2f}")
            return True
        else:
            print("❌ No market data retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Market data test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Alpaca API Integration")
    print("=" * 40)
    
    connection_ok = test_alpaca_connection()
    market_data_ok = test_market_data()
    
    if connection_ok and market_data_ok:
        print("
✅ All Alpaca tests passed!")
    else:
        print("
❌ Some Alpaca tests failed!")

