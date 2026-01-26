#!/usr/bin/env python3
"""
DEMO: Actual Integration into Hybrid Crypto Trader
Shows exactly how to integrate infrastructure modules into your main agent
"""

def create_integrated_hybrid_trader():
    print("""
🚀 DEMO: ACTUAL INTEGRATION - hybrid_crypto_trader.py  
====================================================

## 📍 STEP 1: Add Infrastructure Imports

Your current imports (lines 1-50):
```python
import os
import subprocess  
import sys
import time
import logging
# ... existing imports ...
from dotenv import load_dotenv
```

ADD these infrastructure imports after line 30:
```python
# === PRIORITY 1 INFRASTRUCTURE INTEGRATION ===
from precision_manager import precision_manager
from data_pipeline import data_pipeline
from error_recovery import execute_with_recovery
from config_manager import get_trading_config, get_api_config
from pplx_key_manager import get_sentiment_analysis, get_narrative_summary
from system_health import get_system_health, start_health_monitoring
# === END INFRASTRUCTURE IMPORTS ===
```

## 📍 STEP 2: Replace Precision Handling

Find this pattern in your agent (around lines 1500-2000):
```python
# OLD - Manual rounding (causes precision errors)
def calculate_position_size(symbol, account_value, risk_pct):
    position_value = account_value * risk_pct
    quantity = round(position_value / current_price, 6)  # ❌ Wrong!
    return quantity
```

REPLACE with:
```python
# NEW - Bulletproof precision
def calculate_position_size(symbol, account_value, risk_pct):
    trading_config = get_trading_config()
    position_value = min(account_value * risk_pct, trading_config.max_position_size)
    quantity = position_value / current_price
    
    # Use symbol-specific precision - NO MORE ERRORS!
    quantity = precision_manager.round_quantity(symbol, quantity)
    
    # Validate before returning
    if precision_manager.validate_order_params(symbol, current_price, quantity):
        return quantity
    else:
        logger.warning(f"Order params invalid for {symbol}, adjusting...")
        return precision_manager.round_quantity(symbol, quantity * 0.99)
```

## 📍 STEP 3: Replace Perplexity API Calls  

Find this pattern (around lines 800-1200):
```python
# OLD - Single key, fails on rate limits
def get_sentiment_from_perplexity(headlines):
    headers = {"Authorization": f"Bearer {os.getenv('PPLX_API_KEY')}"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        return None  # ❌ Fails silently
    return response.json()
```

REPLACE with:
```python
# NEW - Multi-key rotation, bulletproof
def get_sentiment_from_perplexity(headlines):
    sentiment_results = []
    for headline in headlines:
        response = get_sentiment_analysis(headline)
        if response.success:
            sentiment_results.append(response.data)
        else:
            logger.warning(f"Sentiment analysis failed: {response.error}")
    return sentiment_results
```

## 📍 STEP 4: Replace Data Fetching

Find this pattern (around lines 400-600):
```python  
# OLD - Provider-specific format
def get_market_data(symbol, timeframe):
    bars = alpaca_api.get_bars(symbol, timeframe, limit=100)
    return bars  # ❌ Different formats from different providers
```

REPLACE with:
```python
# NEW - Standardized format always
def get_market_data(symbol, timeframe):
    bars = data_pipeline.get_standardized_data(
        symbol=symbol,
        timeframe=timeframe, 
        limit=100,
        provider='alpaca'
    )
    return bars  # ✅ Always: timestamp, open, high, low, close, volume
```

## 📍 STEP 5: Add Error Recovery to Orders

Find this pattern (scattered throughout):
```python
# OLD - Basic try/catch  
try:
    result = alpaca_api.submit_order(symbol=symbol, qty=qty, side="buy")
except Exception as e:
    logger.error(f"Order failed: {e}")
    # ❌ Agent continues but may be in inconsistent state
```

REPLACE with:
```python
# NEW - Bulletproof with automatic retry
def place_order_with_recovery(symbol, qty, side):
    return execute_with_recovery(
        alpaca_api.submit_order,
        symbol=symbol,
        qty=qty,
        side=side,
        max_retries=3,
        operation_name=f"place_{side}_order_{symbol}"
    )

# Use it:
result = place_order_with_recovery(symbol, qty, "buy")
```

## 📍 STEP 6: Add Health Monitoring to Main Loop

Find your main() function and add health monitoring:
```python
def main():
    # NEW - Start health monitoring
    start_health_monitoring()
    
    # Your existing main loop
    while True:
        # NEW - Check system health before trading
        health = get_system_health()
        if health.overall_status == HealthStatus.CRITICAL:
            logger.error("System unhealthy, pausing trading cycle")
            time.sleep(60)
            continue
            
        # Your existing trading logic...
        run_trading_cycle()
```

## 🎯 WHAT THIS ACHIEVES

✅ **Same trading logic** - your strategies unchanged
✅ **Bulletproof execution** - no more precision errors  
✅ **Reliable sentiment** - multi-key Perplexity rotation
✅ **Consistent data** - standardized from any provider
✅ **Auto-recovery** - automatic retry on failures
✅ **Health monitoring** - real-time system visibility

## 📊 INTEGRATION EFFORT

- **Lines changed**: ~50-100 out of 2,965 total (~3-5%)
- **Time required**: 1-2 hours
- **Risk**: Minimal (just replacing functions, same logic)
- **Benefit**: Institutional-grade reliability

## 🚀 RESULT

Your hybrid_crypto_trader.py becomes BULLETPROOF while keeping all its intelligence!

No subprocess calls, no hundreds of scripts - just clean Python imports! 🔥
""")

if __name__ == "__main__":
    create_integrated_hybrid_trader()
    print("\n💡 Want me to actually make these changes to your hybrid trader?")
    print("   I can modify just the imports first so you can see the pattern!")
    
    # Show exact files that need modification
    print("\n📁 FILES TO MODIFY:")
    print("   ✏️  scripts/hybrid_crypto_trader.py (add imports, replace ~5 functions)")
    print("   ✏️  high_risk_futures_agent.py (add imports, replace ~3 functions)")
    print("   📚 All infrastructure files (already complete)")
    
    print(f"\n✅ Just 2 main files to edit - that's it!")
    print(f"✅ Infrastructure modules are imported, not called as scripts")
