#!/usr/bin/env python3
"""
REAL Position Closer - Actually close positions on Binance Testnet
"""

import os
import sys
import json
import time
import hmac
import hashlib
import requests
from datetime import datetime
from urllib.parse import urlencode

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Binance testnet credentials  
API_KEY = os.getenv('BINANCE_TESTNET_API_KEY')
SECRET_KEY = os.getenv('BINANCE_TESTNET_SECRET_KEY')
BASE_URL = 'https://testnet.binancefuture.com'

def create_signature(query_string, secret_key):
    """Create HMAC SHA256 signature"""
    return hmac.new(
        secret_key.encode('utf-8'),
        query_string.encode('utf-8'), 
        hashlib.sha256
    ).hexdigest()

def get_account_info():
    """Get account information"""
    endpoint = '/fapi/v2/account'
    timestamp = int(time.time() * 1000)
    query_string = f'timestamp={timestamp}'
    signature = create_signature(query_string, SECRET_KEY)
    
    headers = {'X-MBX-APIKEY': API_KEY}
    url = f'{BASE_URL}{endpoint}?{query_string}&signature={signature}'
    
    try:
        response = requests.get(url, headers=headers)
        return response.json()
    except Exception as e:
        print(f'Error getting account info: {e}')
        return None

def close_position_market_order(symbol, side, quantity):
    """Close position with market order"""
    endpoint = '/fapi/v1/order'
    timestamp = int(time.time() * 1000)
    
    params = {
        'symbol': symbol,
        'side': side, 
        'type': 'MARKET',
        'quantity': quantity,
        'timestamp': timestamp
    }
    
    query_string = urlencode(params)
    signature = create_signature(query_string, SECRET_KEY)
    
    headers = {'X-MBX-APIKEY': API_KEY}
    url = f'{BASE_URL}{endpoint}'
    
    params['signature'] = signature
    
    try:
        response = requests.post(url, headers=headers, params=params)
        return response.json()
    except Exception as e:
        print(f'Error placing order: {e}')
        return None

def main():
    """Close all positions"""
    print('🚨 REAL POSITION CLOSURE - BINANCE TESTNET')
    print('=' * 60)
    
    if not API_KEY or not SECRET_KEY:
        print('❌ Missing API credentials')
        return
    
    # Get account info
    print('🔍 Fetching account information...')
    account = get_account_info()
    
    if not account:
        print('❌ Failed to get account information')
        return
        
    if 'positions' not in account:
        print('❌ No positions data in account response')
        print(f'Response: {account}')
        return
    
    # Find active positions
    active_positions = []
    for pos in account['positions']:
        if float(pos['positionAmt']) != 0:
            active_positions.append(pos)
    
    if not active_positions:
        print('✅ No active positions found')
        return
    
    print(f'📊 Found {len(active_positions)} active positions:')
    for pos in active_positions:
        symbol = pos['symbol']
        size = float(pos['positionAmt'])
        entry_price = float(pos.get('entryPrice', 0))
        pnl = float(pos.get('unRealizedProfit', 0))
        side = 'LONG' if size > 0 else 'SHORT'
        
        print(f'   {symbol}: {side} {abs(size)} @ ${entry_price:.4f} (PnL: ${pnl:.2f})')
    
    # Close all positions
    print(f'\n🔄 Closing {len(active_positions)} positions...')
    
    for pos in active_positions:
        symbol = pos['symbol']
        position_amt = float(pos['positionAmt'])
        
        if position_amt == 0:
            continue
            
        # Determine close side and quantity
        if position_amt > 0:
            close_side = 'SELL'
            quantity = abs(position_amt)
        else:
            close_side = 'BUY'
            quantity = abs(position_amt)
        
        print(f'Closing {symbol} {close_side} {quantity}...')
        
        # Place market order to close
        result = close_position_market_order(symbol, close_side, quantity)
        
        if result and 'orderId' in result:
            print(f'✅ {symbol} close order placed: {result["orderId"]}')
        else:
            print(f'❌ Failed to close {symbol}: {result}')
        
        # Small delay between orders
        time.sleep(0.5)
    
    print('\n✅ Position closure process complete!')
    print('🔍 Checking final positions...')
    
    # Verify closure
    time.sleep(2)
    final_account = get_account_info()
    if final_account and 'positions' in final_account:
        remaining = [p for p in final_account['positions'] if float(p['positionAmt']) != 0]
        if remaining:
            print(f'⚠️ {len(remaining)} positions still open')
            for pos in remaining:
                print(f'   {pos["symbol"]}: {pos["positionAmt"]}')
        else:
            print('🎯 All positions successfully closed!')

if __name__ == '__main__':
    main()
