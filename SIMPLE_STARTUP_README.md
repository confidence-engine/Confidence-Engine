# 🚀 Simple Trading System Scripts (No Admin Required)

These scripts provide a simple way to run the trading system **without requiring admin privileges**.

## 📁 Available Scripts

### `start_simple.sh` - Simple Startup
- ✅ **No admin privileges required**
- ✅ **Real testnet trading** (Alpaca + Binance futures testnet)
- ✅ **Clean startup** with minimal complexity
- ✅ **Safe environment loading**

### `stop_trading.sh` - Stop All Agents
- ✅ **Gracefully stops** all trading processes
- ✅ **Force kill** if needed

### `check_status.sh` - System Status
- ✅ **Real-time status** of all agents
- ✅ **Log file information**
- ✅ **Recent activity** summary

### `start_trading_system_no_admin.sh` - Full Deployment (No Admin)
- ✅ **Complete system** with monitoring
- ✅ **User-space watchdog** (no cron required)
- ✅ **Database logging**
- ✅ **Discord/Telegram notifications**

## 🎯 Quick Start

### 1. Simple Start (Recommended)
```bash
./start_simple.sh
```

### 2. Check Status
```bash
./check_status.sh
```

### 3. View Live Logs
```bash
tail -f hybrid_agent_simple.log futures_agent_simple.log
```

### 4. Stop System
```bash
./stop_trading.sh
```

## 📊 What's Running

When you start the system, you get:

- **Hybrid Agent**: Spot trading on Alpaca testnet
- **Futures Agent**: Leveraged trading on Binance futures testnet
- **Real Testnet Trading**: No simulation, actual testnet platforms
- **Comprehensive Logging**: All activity logged to files

## 🔧 Key Differences from Original

| Feature | Original Script | Simple Scripts |
|---------|----------------|----------------|
| Admin Required | ❌ YES (cron jobs) | ✅ NO |
| Complexity | High | Low |
| Monitoring | Cron-based | Optional user-space |
| Database | Full setup | Optional |
| Notifications | Full setup | Optional |
| Startup Time | Slow | Fast |

## 🛡️ Configuration

The scripts automatically:
- ✅ Load `.env` configuration safely
- ✅ Set `TB_PAPER_TRADING=0` (real testnet)
- ✅ Set `TB_TRADER_OFFLINE=0` (online mode)
- ✅ Set `TB_NO_TRADE=0` (trading enabled)

## 📋 Monitoring

### Real-time Process Check
```bash
ps aux | grep -E '(hybrid|futures)' | grep -v grep
```

### Live Logs
```bash
# Hybrid Agent
tail -f hybrid_agent_simple.log

# Futures Agent  
tail -f futures_agent_simple.log

# Both together
tail -f hybrid_agent_simple.log futures_agent_simple.log
```

### Database Queries (if using full deployment)
```bash
sqlite3 enhanced_trading.db "SELECT * FROM enhanced_trades ORDER BY timestamp DESC LIMIT 10;"
```

## 🚨 Emergency Stop
```bash
# Quick stop
./stop_trading.sh

# Manual stop
pkill -f 'python3.*hybrid_crypto_trader.py'
pkill -f 'python3.*high_risk_futures_agent.py'
```

## 💡 Troubleshooting

### Script Won't Start
1. Check if scripts are executable: `ls -la *.sh`
2. Make executable: `chmod +x *.sh`
3. Check .env file syntax

### Agents Not Running
1. Check logs: `./check_status.sh`
2. View startup logs: `cat hybrid_agent_simple.log`
3. Check for port conflicts: `lsof -i :8080`

### Performance Issues
1. Check system resources: `top`
2. Check log file sizes: `du -h *.log`
3. Restart system: `./stop_trading.sh && ./start_simple.sh`

## 📈 Expected Behavior

- **Hybrid Agent**: Trades spot crypto on Alpaca testnet
- **Futures Agent**: Trades futures on Binance testnet with 25x leverage
- **Real Money**: Uses testnet funds (not real money)
- **Real Platforms**: Actual API connections to testnet environments
- **No Simulation**: All trades execute on real testnet platforms

## 🔄 Automated Restart

For automated restarts without cron, you can use:

```bash
# Simple restart loop (run in screen/tmux)
while true; do
  sleep 21600  # 6 hours
  ./stop_trading.sh
  sleep 10
  ./start_simple.sh
  echo "Restarted at $(date)"
done
```

## 🎯 Why Use These Scripts?

1. **No Admin Privileges**: Works on any system without sudo
2. **Simple & Reliable**: Minimal complexity, fewer failure points
3. **Real Trading**: Uses actual testnet platforms, not simulation
4. **Easy Debugging**: Clear log files and status checking
5. **Quick Startup**: Fast deployment without complex setup

---

**💡 Tip**: Use `./start_simple.sh` for most cases. Only use the full deployment script if you need comprehensive monitoring and database logging.
