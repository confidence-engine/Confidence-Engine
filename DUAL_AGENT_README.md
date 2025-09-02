# Dual Agent Trading System

## Overview
You now have **two independent trading agents** running side-by-side:

1. **🤖 Main Agent (Low-Risk)** - Enhanced hybrid agent with ML gates and adaptive strategies
2. **⚡ Futures Agent (High-Risk)** - Leveraged futures/perpetuals agent with momentum trading

This architecture gives you the best of both worlds: **stable, proven performance** from your main agent plus **high-upside potential** from leveraged futures trading.

## 🚀 Current Status (September 2, 2025)
- **✅ Both Agents Running**: Main agent and futures agent operational
- **✅ Independent Operation**: Separate capital allocation and risk management
- **✅ Real-time Monitoring**: Discord/Telegram notifications active
- **✅ Multi-Platform Support**: Binance Futures (primary), Bybit (backup)
- **✅ Smart Features**: Market regime detection, correlation filtering, trailing stops

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dual Agent System                        │
├─────────────────────────────────────────────────────────────┤
│  🤖 Main Agent (Low-Risk)                                   │
│  • Existing hybrid crypto trader                            │
│  • Spot trading with 20+ assets                             │
│  • Conservative risk management                             │
│  • Stable, proven performance                               │
├─────────────────────────────────────────────────────────────┤
│  ⚡ Futures Agent (High-Risk)                                │
│  • Leveraged futures & perpetuals                           │
│  • 5x-25x leverage on BTC, ETH, SOL                         │
│  • Momentum-based signals                                   │
│  • High-risk, high-reward strategies                        │
├─────────────────────────────────────────────────────────────┤
│  📊 Shared Infrastructure                                    │
│  • Common data sources (Yahoo, Binance, etc.)               │
│  • Unified monitoring dashboard                             │
│  • Independent execution & risk management                  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start Both Agents
```bash
# Start both agents
./dual_agent.sh start

# Check status
./dual_agent.sh status

# View logs
./dual_agent.sh logs
```

### 2. Monitor Performance
```bash
# Real-time dashboard
python3 dual_agent_monitor.py --watch

# Export report
python3 dual_agent_monitor.py --export
```

### 3. Individual Control
```bash
# Control main agent only
./dual_agent.sh main start
./dual_agent.sh main stop
./dual_agent.sh main logs

# Control futures agent only
./dual_agent.sh futures start
./dual_agent.sh futures stop
./dual_agent.sh futures logs
```

## Agent Comparison

| Feature | Main Agent | Futures Agent |
|---------|------------|---------------|
| **Risk Level** | Low | High |
| **Trading Type** | Spot | Futures/Perpetuals |
| **Leverage** | 1x | 5x-25x |
| **Assets** | 20+ crypto | BTC, ETH, SOL |
| **Strategy** | ML + TA signals | Momentum-based |
| **Capital** | Full portfolio | $10k allocation |
| **Goal** | Steady returns | High upside |

## Futures Agent Details

### Strategy
- **Momentum Trading**: 12-hour momentum windows
- **High Leverage**: Up to 25x on strong signals
- **Risk Management**: 5% risk per trade, 20% daily loss limit
- **Position Limits**: Max 3 open positions
- **Exit Rules**: 5% profit target, 3% stop loss, 10% trailing stop

### Configuration
Edit `futures_agent_config.env`:
```bash
# Risk settings
FUTURES_AGENT_CAPITAL=10000          # $10k starting capital
FUTURES_MAX_LEVERAGE=25              # Maximum leverage
FUTURES_RISK_PER_TRADE=0.05          # 5% risk per trade
FUTURES_MAX_DAILY_LOSS=0.20          # 20% max daily loss

# Trading symbols
FUTURES_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT

# Strategy parameters
FUTURES_MOMENTUM_WINDOW=12           # Hours for momentum calc
FUTURES_MIN_MOMENTUM_THRESHOLD=0.02  # 2% min momentum
```

### Supported Platforms
- **Binance Futures Testnet** - Primary platform
- **Bybit Futures Demo** - Alternative platform
- **BitMEX Futures Testnet** - Professional platform
- **Deribit Futures Test** - Options & futures

## Risk Management

### Independent Risk Controls
Each agent has its own risk management:
- **Main Agent**: Conservative position sizing, correlation limits
- **Futures Agent**: High leverage with strict loss limits

### Daily Loss Limits
- **Main Agent**: Portfolio-level VaR limits
- **Futures Agent**: 20% daily loss limit before stopping

### Position Isolation
- Agents trade different instruments
- Independent capital allocation
- Separate position tracking

## Monitoring & Alerts

### Real-Time Dashboard
```bash
python3 dual_agent_monitor.py --watch --interval 30
```

Shows:
- ✅ Agent status (running/stopped)
- 💰 Daily P&L for each agent
- 📊 Trade counts and win rates
- 📈 Open positions
- 🔄 Last activity timestamps

### Log Files
- `main_agent.log` - Main agent activity
- `futures_agent.log` - Futures agent activity
- `futures_trades.json` - Detailed trade records

### Alerts
Both agents send notifications via:
- **Telegram** - Real-time trade alerts
- **Discord** - Performance summaries

## Performance Tracking

### Daily Reports
```bash
python3 dual_agent_monitor.py --export
# Creates: dual_agent_report_YYYYMMDD_HHMMSS.json
```

### Key Metrics
- **Combined P&L**: Total across both agents
- **Risk-Adjusted Returns**: Sharpe ratio, Sortino ratio
- **Win Rates**: By agent and strategy
- **Drawdown Analysis**: Maximum drawdown tracking

## Safety Features

### Circuit Breakers
- **Daily Loss Limits**: Automatic shutdown on excessive losses
- **Position Limits**: Maximum open positions per agent
- **Leverage Caps**: Configurable maximum leverage

### Paper Trading Mode
Both agents support paper trading:
```bash
# Enable paper trading
export TB_PAPER_TRADING=1
export FUTURES_PAPER_TRADING=1
```

### Emergency Stops
```bash
# Stop all agents immediately
./dual_agent.sh stop

# Stop individual agents
./dual_agent.sh main stop
./dual_agent.sh futures stop
```

## Scaling Strategy

### Phase 1: Current Setup (Validation)
- Main agent: Proven spot trading
- Futures agent: $10k paper trading
- Focus: Validate futures strategies

### Phase 2: Live Futures (3 months)
- Futures agent: $10k live trading
- Main agent: Continue spot trading
- Focus: Real P&L validation

### Phase 3: Scale Up (6 months)
- Futures agent: $50k+ capital
- Main agent: Maintain stability
- Focus: Portfolio optimization

## Troubleshooting

### Agent Not Starting
```bash
# Check dependencies
python3 -c "import pandas, requests, dotenv"

# Check configuration
cat .env | grep TB_ENABLE
cat futures_agent_config.env | grep FUTURES
```

### No Trading Signals
```bash
# Check market data
python3 -c "from futures_integration import enhanced_futures_bars; print(enhanced_futures_bars('BTCUSDT', '1h', 5))"

# Check agent logs
./dual_agent.sh logs futures
```

### High Error Rates
```bash
# Restart agents
./dual_agent.sh restart

# Check API connectivity
curl -s https://testnet.binancefuture.com/fapi/v1/ping
```

## Best Practices

### Daily Routine
1. **Morning**: Check dashboard, review overnight performance
2. **Trading Hours**: Monitor real-time via dashboard
3. **Evening**: Export daily report, analyze performance
4. **Weekly**: Review strategy performance, adjust parameters

### Risk Management
1. **Never exceed daily loss limits**
2. **Monitor correlation between agents**
3. **Regular capital rebalancing**
4. **Keep detailed trading journals**

### Performance Review
1. **Track both individual and combined P&L**
2. **Monitor win rates and risk-adjusted returns**
3. **Regular strategy backtesting**
4. **Document all parameter changes**

## Future Enhancements

### Planned Features
- **Multi-timeframe analysis**
- **Machine learning signals**
- **Portfolio optimization**
- **Advanced risk metrics**
- **Automated parameter tuning**

### Integration Opportunities
- **Cross-agent hedging**
- **Dynamic capital allocation**
- **Strategy correlation analysis**
- **Unified reporting system**

---

## 🎯 Your Dual-Agent System is Ready!

You now have:
- ✅ **Main Agent**: Stable, low-risk spot trading
- ✅ **Futures Agent**: High-risk, high-reward futures trading
- ✅ **Independent Operation**: Agents run separately
- ✅ **Unified Monitoring**: Single dashboard for both
- ✅ **Risk Isolation**: Independent capital and risk management
- ✅ **Easy Management**: Simple startup/shutdown scripts

**Start both agents and let them work for you!** 🚀

```bash
# Launch the system
./dual_agent.sh start

# Monitor performance
python3 dual_agent_monitor.py --watch
```
