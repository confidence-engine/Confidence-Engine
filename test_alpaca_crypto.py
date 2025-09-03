#!/usr/bin/env python3

from dotenv import load_dotenv
load_dotenv()

import alpaca

def test_alpaca_crypto():
    print("=== ALPACA CRYPTO TRADING TEST ===")
    
    # Test connection
    rest = alpaca._rest()
    account = rest.get_account()
    print(f"Account Status: {account.status}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")
    print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
    
    # Get available crypto assets
    print("\n=== AVAILABLE CRYPTO ASSETS ===")
    try:
        assets = rest.list_assets()
        crypto_assets = [asset for asset in assets if asset.asset_class.value == 'crypto']
        print(f"Total assets: {len(assets)}")
        print(f"Crypto assets: {len(crypto_assets)}")
        
        if crypto_assets:
            print("\nTradable crypto symbols:")
            for asset in crypto_assets[:15]:  # Show first 15
                print(f"  {asset.symbol:10} - {asset.name:20} - Status: {asset.status}")
    except Exception as e:
        print(f"Error getting assets: {e}")
    
    # Test data access
    print("\n=== TEST CRYPTO DATA ACCESS ===")
    try:
        # Test BTC data
        bars = alpaca.recent_bars('BTC/USD', minutes=30)
        print(f"BTC/USD bars: {len(bars)} rows")
        if len(bars) > 0:
            close_price = bars['close'].iloc[-1]
            print(f"Latest BTC close: ${close_price:,.2f}")
            print(f"Price range: ${bars['low'].min():,.2f} - ${bars['high'].max():,.2f}")
        
        # Test headlines
        headlines = alpaca.latest_headlines('BTC/USD', limit=3)
        print(f"Headlines available: {len(headlines)}")
        if headlines:
            print(f"Latest headline: {headlines[0][:80]}...")
        
        # Test smoke function
        bars_count, heads_count = alpaca.smoke('BTC/USD')
        print(f"Smoke test - Bars: {bars_count}, Headlines: {heads_count}")
        
    except Exception as e:
        print(f"Data access error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test order capabilities
    print("\n=== TRADING CAPABILITIES ===")
    try:
        # Get current positions
        positions = rest.list_positions()
        print(f"Current positions: {len(positions)}")
        for pos in positions:
            print(f"  {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")
        
        # Get pending orders
        orders = rest.list_orders()
        print(f"Pending orders: {len(orders)}")
        
    except Exception as e:
        print(f"Trading info error: {e}")

if __name__ == "__main__":
    test_alpaca_crypto()
