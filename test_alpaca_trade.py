#!/usr/bin/env python3

import sys
sys.path.append('.')
import os
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST
from config import settings

load_dotenv()

def test_alpaca_trading():
    """Test Alpaca trading functionality with a small paper trade"""
    
    # Initialize Alpaca API using the same method as the agents
    api = REST(
        key_id=settings.alpaca_key_id,
        secret_key=settings.alpaca_secret_key,
        base_url=settings.alpaca_base_url
    )
    
    print('🔍 Testing Alpaca connection and trading capability...')
    
    # Check account
    try:
        account = api.get_account()
        print(f'✅ Account connected: ${float(account.buying_power):,.2f} buying power')
        print(f'📊 Portfolio value: ${float(account.portfolio_value):,.2f}')
        print(f'🏦 Cash: ${float(account.cash):,.2f}')
        print(f'📋 Account status: {account.status}')
    except Exception as e:
        print(f'❌ Account connection failed: {e}')
        return False
    
    # Test getting crypto assets
    try:
        assets = api.list_assets(status='active', asset_class='crypto')
        crypto_assets = [asset.symbol for asset in assets if 'USD' in asset.symbol]
        print(f'✅ Found {len(crypto_assets)} crypto assets available')
        print(f'📈 Available: {crypto_assets[:10]}...')
    except Exception as e:
        print(f'❌ Asset listing failed: {e}')
        return False
    
    # Test a small paper trade on BTC/USD
    try:
        # Check current positions first
        positions = api.list_positions()
        print(f'📊 Current positions: {len(positions)}')
        
        # Get latest BTC price using same pattern as agents
        from alpaca import recent_bars
        bars = recent_bars("BTCUSD", 5)
        if len(bars) > 0:
            btc_price = float(bars.iloc[-1]['close'])
            print(f'💰 Current BTC price: ${btc_price:,.2f}')
        else:
            print('❌ Could not get BTC price')
            return False
        
        # Calculate small test order ($50 worth)
        test_amount = 50.0
        test_quantity = round(test_amount / btc_price, 6)
        
        print(f'🧪 Testing paper trade: ${test_amount} = {test_quantity} BTC')
        
        # Place a small market buy order using same method as agents
        order = api.submit_order(
            symbol='BTC/USD',  # Use the format the agents use
            side='buy',
            type='market',
            time_in_force='gtc',
            qty=test_quantity
        )
        
        print(f'✅ Test order submitted: {order.id}')
        print(f'📋 Order status: {order.status}')
        print(f'💱 Order: {order.side} {order.qty} {order.symbol}')
        
        # Wait a moment and check order status
        import time
        time.sleep(3)
        
        updated_order = api.get_order(order.id)
        print(f'🔄 Updated status: {updated_order.status}')
        
        if updated_order.status == 'filled':
            print(f'✅ TEST TRADE SUCCESSFUL!')
            print(f'💰 Filled at: ${float(updated_order.filled_avg_price):.2f}')
            print(f'📊 Quantity: {updated_order.filled_qty}')
            
            # Immediately sell it back to clean up
            sell_order = api.submit_order(
                symbol='BTC/USD',
                side='sell',
                type='market',
                time_in_force='gtc',
                qty=updated_order.filled_qty
            )
            
            print(f'🔄 Reversing trade: {sell_order.id}')
            
            time.sleep(3)
            sell_updated = api.get_order(sell_order.id)
            print(f'📈 Sell order status: {sell_updated.status}')
            
            return True
            
        else:
            print(f'⚠️ Order not filled: {updated_order.status}')
            if updated_order.status in ['pending_new', 'new']:
                # Cancel the order
                api.cancel_order(order.id)
                print('🚫 Order cancelled')
            return False
                
    except Exception as e:
        print(f'❌ Test trade failed: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    print('🏁 Alpaca trading test completed')
    return True
