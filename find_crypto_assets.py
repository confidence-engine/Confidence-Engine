#!/usr/bin/env python3

from dotenv import load_dotenv
load_dotenv()
import alpaca

def find_crypto_assets():
    rest = alpaca._rest()
    assets = rest.list_assets()
    
    print("First asset attributes:")
    first_asset = assets[0]
    attrs = [attr for attr in dir(first_asset) if not attr.startswith('_')]
    print(attrs)
    
    print(f"\nSample asset details:")
    print(f"  Symbol: {first_asset.symbol}")
    print(f"  Name: {first_asset.name}")
    print(f"  Status: {first_asset.status}")
    if hasattr(first_asset, 'asset_class'):
        print(f"  Asset Class: {first_asset.asset_class}")
    
    # Find crypto assets by checking symbol patterns
    crypto_symbols = []
    for asset in assets:
        # Look for crypto patterns
        symbol = asset.symbol
        if ('/' in symbol and 'USD' in symbol) or symbol.endswith('USD'):
            crypto_symbols.append(symbol)
    
    print(f"\nFound {len(crypto_symbols)} potential crypto symbols:")
    for sym in sorted(crypto_symbols):
        print(f"  {sym}")
    
    # Test a few to see if they work
    test_symbols = ['BTC/USD', 'ETH/USD', 'DOGE/USD', 'SOL/USD']
    print(f"\nTesting symbol access:")
    for sym in test_symbols:
        try:
            bars = alpaca.recent_bars(sym, minutes=15)
            print(f"  {sym:8} - OK ({len(bars)} bars)")
        except Exception as e:
            print(f"  {sym:8} - ERROR: {e}")

if __name__ == "__main__":
    find_crypto_assets()
