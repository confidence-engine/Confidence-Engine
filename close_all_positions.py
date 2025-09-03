#!/usr/bin/env python3
"""
Close all Binance futures positions
"""
import requests
import os
import time
import hmac
import hashlib
from dotenv import load_dotenv

load_dotenv()

# Binance testnet credentials
API_KEY = os.getenv('BINANCE_FUTURES_TESTNET_API_KEY')
SECRET_KEY = os.getenv('BINANCE_FUTURES_TESTNET_SECRET_KEY')
BASE_URL = 'https://testnet.binancefuture.com'

def create_signature(query_string, secret_key):
    """Create HMAC SHA256 signature"""
    return hmac.new(
        secret_key.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def close_all_positions():
    """Close all open futures positions"""
    headers = {'X-MBX-APIKEY': API_KEY}
    
    print('🔄 CLOSING ALL FUTURES POSITIONS')
    print('=' * 40)
    
    # Get current positions
    response = requests.get(f'{BASE_URL}/fapi/v2/positionRisk', headers=headers)
    
    if response.status_code != 200:
        print(f"❌ Error getting positions: {response.text}")
        return
    
    positions = response.json()
    active_positions = [pos for pos in positions if float(pos['positionAmt']) != 0]
    
    if not active_positions:
        print("✅ No active positions to close")
        return
    
    print(f"Found {len(active_positions)} active positions to close:")
    
    for pos in active_positions:
        symbol = pos['symbol']
        position_amt = float(pos['positionAmt'])
        side = 'SELL' if position_amt > 0 else 'BUY'
        quantity = abs(position_amt)
        
        print(f"\n📍 Closing {symbol}:")
        print(f"  Position: {position_amt}")
        print(f"  Side: {side}")
        print(f"  Quantity: {quantity}")
        
        # Create order parameters
        timestamp = int(time.time() * 1000)
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': quantity,
            'reduceOnly': 'true',
            'timestamp': timestamp
        }
        
        # Create query string for signature
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = create_signature(query_string, SECRET_KEY)
        params['signature'] = signature
        
        # Place close order
        close_response = requests.post(
            f'{BASE_URL}/fapi/v1/order',
            headers=headers,
            data=params
        )
        
        if close_response.status_code == 200:
            order_data = close_response.json()
            print(f"  ✅ Order placed: {order_data.get('orderId')}")
        else:
            print(f"  ❌ Error closing position: {close_response.text}")
        
        # Small delay between orders
        time.sleep(0.5)
    
    print(f"\n✅ Finished closing positions")

if __name__ == "__main__":
    close_all_positions()
