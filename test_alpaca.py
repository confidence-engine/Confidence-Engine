#!/usr/bin/env python3
import os
from dotenv import load_dotenv
load_dotenv()
from alpaca_trade_api import REST

api = REST(
    key_id=os.getenv('ALPACA_API_KEY_ID'),
    secret_key=os.getenv('ALPACA_API_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
)

try:
    account = api.get_account()
    print(f'Account ID: {account.id}')
    print(f'Equity: ${account.equity}')
    print(f'Buying Power: ${account.buying_power}')
    print(f'Account Status: {account.status}')
    print('✅ Alpaca connection successful - using new account')
except Exception as e:
    print(f'❌ Alpaca connection failed: {e}')
