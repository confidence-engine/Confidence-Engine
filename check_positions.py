#!/usr/bin/env python3
import os
import requests
import time
import hmac
import hashlib
from dotenv import load_dotenv

load_dotenv()

# Binance testnet credentials
api_key = os.getenv('BINANCE_TESTNET_API_KEY')
secret_key = os.getenv('BINANCE_TESTNET_SECRET_KEY')

print('🔍 CHECKING BINANCE FUTURES POSITIONS')
print('=' * 45)

def create_signature(query_string, secret_key):
    return hmac.new(secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

def make_binance_request(endpoint, params=None):
    base_url = 'https://testnet.binancefuture.com'
    url = base_url + endpoint

    if params is None:
        params = {}

    timestamp = str(int(time.time() * 1000))
    params['timestamp'] = timestamp

    # Create query string
    query_string = '&'.join([f'{k}={v}' for k, v in params.items()])

    # Create signature
    signature = create_signature(query_string, secret_key)
    params['signature'] = signature

    headers = {
        'X-MBX-APIKEY': api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        print(f'GET {endpoint}: HTTP {response.status_code}')

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f'❌ HTTP Error: {response.text}')
            return None

    except Exception as e:
        print(f'❌ Exception: {str(e)}')
        return None

# Check positions
print()
print('1. Checking positions...')
positions_data = make_binance_request('/fapi/v2/positionRisk')
if positions_data:
    active_positions = []
    for pos in positions_data:
        position_amt = float(pos.get('positionAmt', 0))
        if position_amt != 0:
            entry_price = float(pos.get('entryPrice', 0))
            mark_price = float(pos.get('markPrice', 0))
            pnl_pct = 0
            if entry_price > 0:
                if position_amt > 0:  # LONG
                    pnl_pct = (mark_price - entry_price) / entry_price * 100
                else:  # SHORT
                    pnl_pct = (entry_price - mark_price) / entry_price * 100

            active_positions.append({
                'symbol': pos.get('symbol', ''),
                'side': 'LONG' if position_amt > 0 else 'SHORT',
                'size': abs(position_amt),
                'entry_price': entry_price,
                'mark_price': mark_price,
                'pnl': float(pos.get('unRealizedProfit', 0)),
                'pnl_pct': pnl_pct
            })

    if active_positions:
        print(f'📊 Active positions: {len(active_positions)}')
        for pos in active_positions:
            pnl_symbol = '+' if pos['pnl'] >= 0 else ''
            print(f'  {pos["symbol"]} {pos["side"]}: {pos["size"]} @ ${pos["entry_price"]:.2f} | P&L: {pnl_symbol}${pos["pnl"]:.2f} ({pos["pnl_pct"]:.2f}%)')
    else:
        print('📊 No active positions found')
else:
    print('❌ Failed to get positions')
