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

def close_position(symbol, side, quantity):
    """Close a position by placing an opposite order"""
    print(f'\n🔄 Closing {symbol} {side} position: {quantity}')

    # Determine opposite side
    close_side = 'SELL' if side == 'LONG' else 'BUY'

    # Prepare order parameters
    order_params = {
        'symbol': symbol,
        'side': close_side,
        'type': 'MARKET',
        'quantity': str(abs(quantity)),
        'timestamp': str(int(time.time() * 1000)),
        'recvWindow': 5000
    }

    # Create signature
    query_string = '&'.join([f'{k}={v}' for k, v in order_params.items()])
    signature = create_signature(query_string, secret_key)
    order_params['signature'] = signature

    headers = {'X-MBX-APIKEY': api_key}

    try:
        response = requests.post(
            'https://testnet.binancefuture.com/fapi/v1/order',
            headers=headers,
            data=order_params,
            timeout=10
        )

        if response.status_code == 200:
            order_data = response.json()
            print(f'✅ Successfully closed {symbol} position')
            print(f'   Order ID: {order_data.get("orderId", "N/A")}')
            print(f'   Status: {order_data.get("status", "N/A")}')
            return True
        else:
            print(f'❌ Failed to close {symbol}: {response.text}')
            return False

    except Exception as e:
        print(f'❌ Exception closing {symbol}: {str(e)}')
        return False

# Check positions
print()
print('1. Checking positions...')
positions_data = make_binance_request('/fapi/v2/positionRisk')
if positions_data:
    active_positions = []
    underwater_positions = []

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

            position_info = {
                'symbol': pos.get('symbol', ''),
                'side': 'LONG' if position_amt > 0 else 'SHORT',
                'size': abs(position_amt),
                'entry_price': entry_price,
                'mark_price': mark_price,
                'pnl': float(pos.get('unRealizedProfit', 0)),
                'pnl_pct': pnl_pct
            }

            active_positions.append(position_info)

            # Check if position is underwater (beyond stop loss threshold)
            if pnl_pct <= -3.0:  # 3% stop loss
                underwater_positions.append(position_info)

    if active_positions:
        print(f'📊 Active positions: {len(active_positions)}')
        for pos in active_positions:
            pnl_symbol = '+' if pos['pnl'] >= 0 else ''
            print(f'  {pos["symbol"]} {pos["side"]}: {pos["size"]} @ ${pos["entry_price"]:.2f} | P&L: {pnl_symbol}${pos["pnl"]:.2f} ({pos["pnl_pct"]:.2f}%)')

        # Handle underwater positions
        if underwater_positions:
            print(f'\n🚨 Found {len(underwater_positions)} underwater positions (beyond 3% stop loss):')
            for pos in underwater_positions:
                print(f'  ❌ {pos["symbol"]} {pos["side"]}: {pos["pnl_pct"]:.2f}% loss')

            print(f'\n🔄 Closing underwater positions...')
            for pos in underwater_positions:
                success = close_position(pos['symbol'], pos['side'], pos['size'])
                if success:
                    print(f'✅ Closed {pos["symbol"]} position')
                else:
                    print(f'❌ Failed to close {pos["symbol"]} position')
        else:
            print('\n✅ No underwater positions found (all above 3% stop loss threshold)')
    else:
        print('📊 No active positions found')
else:
    print('❌ Failed to get positions')
