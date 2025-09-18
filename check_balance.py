from alpaca import _rest

api = _rest()
account = api.get_account()
print(f"Cash: ${float(account.cash):.2f}")
print(f"Equity: ${float(account.equity):.2f}")
print(f"Buying power: ${float(account.buying_power):.2f}")
print(f"Status: {account.status}")
print(f"Trading blocked: {account.trading_blocked}")
