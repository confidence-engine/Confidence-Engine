# Futures & Perpetuals Trading Platform

## Overview
This platform provides **free futures and perpetuals trading** capabilities for your crypto trading agent. Unlike Alpaca's spot-only trading, this system supports:

- **Futures Contracts** (BTC futures, quarterly contracts)
- **Perpetual Swaps** (BTC-PERP, ETH-PERP, etc.)
- **High Leverage** (up to 125x on some platforms)
- **Multi-Asset Support** (BTC, ETH, SOL, ADA, DOT, etc.)
- **Paper Trading Mode** (free testing without real money)
- **Professional Tools** (advanced order types, position management)

## 🚀 Current Status (September 2, 2025)
- **✅ High-Risk Futures Agent**: Running live with real Binance testnet integration
- **✅ Platform**: Binance Futures testnet with HMAC SHA256 authentication 
- **✅ Smart Leverage**: Dynamic risk-reward based calculations (up to 25x cap)
- **✅ Risk Management**: $100 margin cap per trade, volatility adjustments
- **✅ Position Management**: Market regime detection, correlation filtering
- **✅ Order Precision**: Fixed quantity precision issues for reliable execution
- **✅ Notifications**: Real-time Discord/Telegram alerts with trade confirmations
- **✅ Database Tracking**: SQLite auto-commit, GitHub integration active

## Supported Platforms

### 1. Binance Futures Testnet (PRIMARY - ACTIVE)
- **Type**: Futures & Perpetuals
- **Leverage**: Up to 125x (capped at 25x for safety)
- **Assets**: 20 blue chip pairs (BTCUSDT, ETHUSDT, SOLUSDT, ADAUSDT, etc.)
- **Features**: Real API integration, dynamic precision handling
- **Status**: ✅ OPERATIONAL - Live trading with real testnet orders
- **Recent Orders**: ADAUSDT x25, ENJUSDT x25 successfully placed
- **Cost**: Free (testnet with real API calls)

### 2. Bybit Futures Demo (BACKUP - DISABLED)
- **Type**: Futures & Perpetuals  
- **Status**: ❌ DISABLED - API authentication issues resolved by focusing on Binance
- **Reason**: Simplified to single-platform approach for reliability

### 3. BitMEX Futures Testnet (NOT IMPLEMENTED)
- **Status**: ❌ NOT IMPLEMENTED - Focus on Binance stability first

### 4. Deribit Futures Test (NOT IMPLEMENTED)
- **Status**: ❌ NOT IMPLEMENTED - Focus on Binance stability first

## Setup Instructions

### 1. Environment Variables
Add these to your `.env` file:

```bash
# Enable futures trading
TB_ENABLE_FUTURES_TRADING=1

# Choose your preferred platform
TB_FUTURES_PLATFORM=binance  # Options: binance, bybit, bitmex, deribit

# Risk management
TB_MAX_LEVERAGE=10           # Maximum leverage to use
TB_PAPER_TRADING=1           # 1 for paper trading, 0 for live

# Optional API keys (for authenticated access)
BINANCE_TESTNET_API_KEY=your_key_here
BINANCE_TESTNET_SECRET_KEY=your_secret_here
BYBIT_TESTNET_API_KEY=your_key_here
BYBIT_TESTNET_SECRET_KEY=your_secret_here
BITMEX_TESTNET_API_KEY=your_key_here
BITMEX_TESTNET_SECRET_KEY=your_secret_here
DERIBIT_TEST_CLIENT_ID=your_id_here
DERIBIT_TEST_CLIENT_SECRET=your_secret_here
```

### 2. Install Dependencies
```bash
pip install requests pandas numpy
```

### 3. Test the Platform
```bash
# Run the demo
python3 futures_paper_trading_demo.py

# Test integration
python3 futures_integration.py
```

## Usage Examples

### Basic Data Fetching
```python
from futures_integration import enhanced_futures_bars

# Get BTC futures data
btc_data = enhanced_futures_bars("BTCUSDT", "1h", 100)
print(f"Got {len(btc_data)} bars")
```

### Position Sizing
```python
from futures_integration import calculate_futures_position

# Calculate position for $10k capital, 2% risk
pos_info = calculate_futures_position("BTCUSDT", 10000.0, 0.02)
print(f"Position: ${pos_info['position_value']:.2f} at {pos_info['leverage_used']}x")
```

### Trading Execution
```python
from futures_integration import execute_futures_trade

# Execute a trade
trade = execute_futures_trade("BTCUSDT", "buy", pos_info)
print(f"Order placed: {trade['order_id']}")
```

### Portfolio Management
```python
from futures_integration import get_futures_status

# Get portfolio status
status = get_futures_status()
print(f"Balance: ${status['balance']['total_balance']:.2f}")
```

## Integration with Your Agent

### Replace Alpaca Functions
Instead of using Alpaca's spot-only functions, use these drop-in replacements:

```python
# Old (Alpaca spot only)
bars = alpaca.get_bars("BTCUSD", timeframe, limit)

# New (Futures & Perpetuals)
bars = enhanced_futures_bars("BTCUSDT", timeframe, limit)
```

### Enhanced Trading Agent
Your agent can now trade:
- **Spot**: BTC/USD (Alpaca)
- **Futures**: BTCUSDT futures contracts
- **Perpetuals**: BTC-PERPETUAL perpetual swaps
- **Multi-asset**: All major crypto assets

### Risk Management
The platform includes:
- **Kelly Criterion** position sizing
- **Volatility-adjusted** risk management
- **Leverage limits** to prevent over-leveraging
- **Portfolio diversification** across assets

## Advanced Features

### Multi-Asset Strategy
```python
# Analyze multiple assets
symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
for symbol in symbols:
    data = enhanced_futures_bars(symbol, '1h', 50)
    # Implement your strategy logic
```

### Leverage Management
```python
# Conservative leverage (2-5x)
pos_conservative = calculate_futures_position("BTCUSDT", capital, risk_pct=0.01)

# Aggressive leverage (10-25x)
pos_aggressive = calculate_futures_position("BTCUSDT", capital, risk_pct=0.05)
```

### Platform Switching
```python
from futures_trading_platform import switch_futures_platform

# Switch to different platform
switch_futures_platform("bybit")  # For different assets/features
```

## Performance Comparison

| Feature | Alpaca | Futures Platform |
|---------|--------|------------------|
| **Trading Type** | Spot only | Futures & Perpetuals |
| **Leverage** | 1x | Up to 125x |
| **Assets** | Limited crypto | Full crypto universe |
| **Cost** | Subscription | Free (paper trading) |
| **Data Frequency** | Standard | High-frequency |
| **Order Types** | Basic | Advanced (limit, stop, etc.) |

## Demo Results
Running the demo typically shows:
- ✅ Data availability across 5+ assets
- ✅ Position sizing calculations
- ✅ Simulated trades with leverage
- ✅ Portfolio status tracking
- ✅ Multi-asset strategy suggestions

## Troubleshooting

### No Data Available
- Check internet connection
- Verify platform availability
- Try different symbol formats (BTCUSDT vs BTC-PERP)

### Platform Not Available
- Some platforms may have downtime
- Switch to alternative platform
- Check platform status pages

### API Rate Limits
- Free tiers have rate limits
- Implement retry logic with backoff
- Use multiple platforms for redundancy

## Next Steps

1. **Run the Demo**: `python3 futures_paper_trading_demo.py`
2. **Configure Environment**: Update your `.env` file
3. **Integrate with Agent**: Replace Alpaca calls with futures functions
4. **Test Strategies**: Paper trade your algorithms
5. **Go Live**: Switch to real trading when ready

## Security Notes

- All platforms are **free demo/testnet** environments
- No real money is at risk during paper trading
- API keys are optional (anonymous access for basic features)
- Data is fetched from official exchange APIs
- All trades are simulated in paper trading mode

---

**🎯 Result**: Your trading agent now supports professional-grade futures and perpetuals trading with leverage, multiple assets, and advanced risk management - all for free through paper trading platforms!
