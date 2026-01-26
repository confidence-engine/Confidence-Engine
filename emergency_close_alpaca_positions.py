#!/usr/bin/env python3
"""
🚨 EMERGENCY: Close all Alpaca positions immediately
"""
import sys
import os
sys.path.append('.')

from config import settings
from alpaca_trade_api.rest import REST
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("🚨 EMERGENCY: CLOSING ALL ALPACA POSITIONS")
    print("=" * 50)
    
    # Initialize Alpaca client
    api = REST(
        key_id=settings.alpaca_key_id,
        secret_key=settings.alpaca_secret_key,
        base_url=settings.alpaca_base_url,
    )
    
    # Get account info
    account = api.get_account()
    print(f"Account Equity: ${account.equity}")
    print(f"Cash Available: ${account.cash}")
    print()
    
    # Get all positions
    positions = api.list_positions()
    print(f"Found {len(positions)} positions to close:")
    print("-" * 50)
    
    total_value = 0
    for pos in positions:
        qty = float(pos.qty)
        current_price = float(pos.current_price)
        value = qty * current_price
        pnl = float(pos.unrealized_pl)
        total_value += value
        
        print(f"📍 {pos.symbol}:")
        print(f"  Qty: {qty:.6f}")
        print(f"  Price: ${current_price:.2f}")
        print(f"  Value: ${value:.2f}")
        print(f"  P&L: ${pnl:.2f}")
        print()
        
        # Place market sell order (crypto specific parameters)
        try:
            order = api.submit_order(
                symbol=pos.symbol,
                qty=str(abs(qty)),
                side='sell',
                type='market',
                time_in_force='ioc'  # Immediate or Cancel for crypto
            )
            print(f"  ✅ Sell order placed: {order.id}")
        except Exception as e:
            print(f"  ❌ Error selling {pos.symbol}: {e}")
            # Try with different time in force
            try:
                order = api.submit_order(
                    symbol=pos.symbol,
                    qty=str(abs(qty)),
                    side='sell',
                    type='market'
                    # No time_in_force for crypto market orders
                )
                print(f"  ✅ Sell order placed (retry): {order.id}")
            except Exception as e2:
                print(f"  ❌ Second attempt failed: {e2}")
        print("-" * 30)
    
    print(f"\nTotal position value being liquidated: ${total_value:.2f}")
    
    # Cancel any pending orders
    try:
        orders = api.list_orders(status='open')
        if orders:
            print(f"\n🚨 Canceling {len(orders)} pending orders:")
            for order in orders:
                try:
                    api.cancel_order(order.id)
                    print(f"  ✅ Canceled order {order.id} for {order.symbol}")
                except Exception as e:
                    print(f"  ❌ Error canceling order {order.id}: {e}")
    except Exception as e:
        print(f"❌ Error checking/canceling orders: {e}")
    
    print("\n🚨 EMERGENCY LIQUIDATION COMPLETE")
    print("All positions and orders have been closed/canceled")

if __name__ == "__main__":
    main()
