# Multi-Source Data Provider for Paper Trading

This system provides **free alternatives to Alpaca** for paper trading and testing your crypto trading agent. Instead of paying for Alpaca's data feeds, you can use multiple free data sources with automatic failover.

## 🎯 What This Solves

- **Cost Reduction**: No API fees for testing and development
- **Redundancy**: Multiple data sources prevent single points of failure
- **Flexibility**: Easy switching between data providers
- **Paper Trading**: Perfect for testing strategies without live trading costs

## 📊 Available Free Data Sources

| Provider | Data Type | Frequency | Cost | Status |
|----------|-----------|-----------|------|--------|
| **Yahoo Finance** | Stocks & Crypto | 1min - Daily | Free | ✅ Active |
| **Binance API** | Crypto Only | 1min - Daily | Free | ✅ Active |
| **CoinGecko** | Crypto Only | 1min - Daily | Free | ✅ Active |
| **Alpha Vantage** | Stocks & Crypto | 1min - Daily | Free API Key | 🔧 Optional |

## 🚀 Quick Start

### 1. Enable Multi-Source Data Provider

Edit your `.env` file:

```bash
# Enable free data sources
TB_ENABLE_MULTI_SOURCE_DATA=1

# Choose your preferred source (yahoo, binance, coingecko, alphavantage)
TB_PREFERRED_DATA_SOURCE=yahoo

# Enable automatic failover
TB_DATA_SOURCE_FAILOVER=1

# Optional: Get free Alpha Vantage API key from https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=your_free_api_key_here
```

### 2. Switch to Paper Trading Mode

For paper trading (recommended for testing):

```bash
# Disable live trading
TB_NO_TRADE=1

# Keep data fetching enabled
TB_OFFLINE=0
```

### 3. Test the Integration

```bash
python3 enhanced_data_integration.py
```

You should see:
```
✅ Using multi-source data (yahoo) for BTC/USD
✅ Successfully fetched 44 bars for BTC/USD
```

## 🔧 Integration with Your Code

### Replace Alpaca Calls

**Before (using Alpaca):**
```python
from alpaca import recent_bars, latest_headlines

bars = recent_bars("BTC/USD", minutes=120)
news = latest_headlines("BTC/USD", limit=10)
```

**After (using free sources):**
```python
from enhanced_data_integration import enhanced_recent_bars, enhanced_latest_headlines

bars = enhanced_recent_bars("BTC/USD", minutes=120)
news = enhanced_latest_headlines("BTC/USD", limit=10)
```

### Automatic Fallback

The system automatically:
1. Tries your preferred data source first
2. Falls back to other free sources if the first fails
3. Uses Alpaca as final fallback (if available)
4. Never fails - always returns data

## 📈 Data Quality Comparison

| Aspect | Alpaca | Yahoo Finance | Binance | CoinGecko |
|--------|--------|---------------|---------|-----------|
| **Cost** | Paid | Free | Free | Free |
| **Crypto Coverage** | 20+ assets | 1000+ assets | 1000+ assets | 1000+ assets |
| **Frequency** | 1min | 1min | 1min | Hourly |
| **Historical Data** | 2+ years | 2+ years | Limited | 1+ year |
| **Real-time** | Yes | Delayed 15min | Yes | Delayed |
| **News** | Limited | None | None | None |

## 🎛️ Configuration Options

### Environment Variables

```bash
# Core settings
TB_ENABLE_MULTI_SOURCE_DATA=1        # Enable/disable multi-source
TB_PREFERRED_DATA_SOURCE=yahoo       # Preferred provider
TB_DATA_SOURCE_FAILOVER=1            # Enable failover

# Provider-specific
ALPHA_VANTAGE_API_KEY=               # For Alpha Vantage (optional)

# Trading mode
TB_NO_TRADE=1                        # Paper trading mode
TB_OFFLINE=0                         # Enable data fetching
```

### Supported Symbols

All major crypto pairs are supported:
- BTC/USD, ETH/USD, SOL/USD, LINK/USD
- LTC/USD, BCH/USD, UNI/USD, AAVE/USD
- AVAX/USD, DOT/USD, MATIC/USD
- And 20+ more crypto assets

## 🧪 Testing & Validation

### Run Integration Tests

```bash
# Test all data sources
python3 enhanced_data_integration.py

# Test specific symbol
python3 -c "
from enhanced_data_integration import enhanced_recent_bars
data = enhanced_recent_bars('BTC/USD', 60)
print(f'Got {len(data)} bars, latest price: ${data[\"close\"].iloc[-1]:.2f}')
"
```

### Monitor Data Quality

The system logs which data source is being used:
```
✅ Using multi-source data (yahoo) for BTC/USD
🏦 Using Alpaca for BTC/USD (fallback)
```

## 🔄 Switching Between Modes

### Development/Paper Trading Mode
```bash
TB_ENABLE_MULTI_SOURCE_DATA=1
TB_NO_TRADE=1
TB_PREFERRED_DATA_SOURCE=yahoo
```

### Production/Live Trading Mode
```bash
TB_ENABLE_MULTI_SOURCE_DATA=0  # Use Alpaca
TB_NO_TRADE=0                  # Enable live trading
```

### Hybrid Mode (Recommended)
```bash
TB_ENABLE_MULTI_SOURCE_DATA=1  # Free sources for analysis
TB_NO_TRADE=0                  # But use Alpaca for execution
```

## 🚨 Important Notes

1. **Data Quality**: Free sources may have slight delays or limited historical data
2. **Rate Limits**: Each provider has rate limits (handled automatically)
3. **News**: Most free sources don't provide news headlines
4. **Real-time**: Yahoo Finance has 15-minute delays for real-time data
5. **Fallback**: Alpaca is always available as final fallback

## 🛠️ Troubleshooting

### Common Issues

**"Multi-Source Enabled: ❌"**
- Check `TB_ENABLE_MULTI_SOURCE_DATA=1` in `.env`
- Restart your Python session

**"No data received"**
- Try different data source: `TB_PREFERRED_DATA_SOURCE=binance`
- Check internet connection
- Some symbols may not be available on all platforms

**"Import errors"**
- Install required packages: `pip install yfinance requests`
- For Alpha Vantage: Get free API key

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Metrics

The system tracks:
- Data source success rates
- Response times
- Failover frequency
- Data quality metrics

Check status anytime:
```bash
python3 enhanced_data_integration.py
```

## 🎯 Use Cases

### Perfect For:
- ✅ Paper trading and strategy testing
- ✅ Development and debugging
- ✅ Backtesting with free data
- ✅ Learning and experimentation
- ✅ Cost-conscious operations

### Less Ideal For:
- ❌ High-frequency trading (< 1 minute)
- ❌ Real-time execution (use Alpaca)
- ❌ Advanced news sentiment analysis
- ❌ Production systems requiring guaranteed uptime

## 🚀 Next Steps

1. **Enable the system**: Set `TB_ENABLE_MULTI_SOURCE_DATA=1`
2. **Test with your agent**: Run your trading scripts
3. **Monitor performance**: Check logs for data source usage
4. **Optimize settings**: Choose the best data source for your needs
5. **Scale up**: Add more assets and test more strategies

---

**🎉 You're now using free data sources for paper trading!**

Your trading agent can now test strategies, validate signals, and develop algorithms **without any API costs**. The multi-source system ensures reliability while the automatic failover prevents disruptions.
