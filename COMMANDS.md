# 🚀 Enhanced Dual Agent System - Complete Command Reference
# =====================================================================

## 🚨 CRITICAL FIX NOTICE - READ FIRST

**MAJOR BUG FIXED:** The hybrid crypto trader was missing ALL ENTRY LOGIC in the main loop! 
This critical bug meant the system could only exit positions but never enter new ones.

**Status:** ✅ FIXED - Entry logic now properly implemented with intelligent TP/SL integration.

**Test immediately with paper trading to verify the fix works:**
```bash
export TB_NO_TRADE=1  
./dual_agent.sh start
```

---

## 🎯 MAIN OPERATIONS

### Start/Stop/Monitor System
```bash
# Start both enhanced agents with intelligent TP/SL
./dual_agent.sh start

# Stop both agents
./dual_agent.sh stop

# Restart both agents
./dual_agent.sh restart

# Check agent status
./dual_agent.sh status

# Monitor logs
./dual_agent.sh logs          # Both agents
./dual_agent.sh logs main     # Hybrid trader only
./dual_agent.sh logs futures  # Futures agent only
```

### Individual Agent Control
```bash
# Hybrid Trader (Main Agent)
./dual_agent.sh main start    # Start only hybrid trader
./dual_agent.sh main stop     # Stop only hybrid trader
./dual_agent.sh main logs     # Show hybrid trader logs

# Futures Agent
./dual_agent.sh futures start # Start only futures agent
./dual_agent.sh futures stop  # Stop only futures agent
./dual_agent.sh futures logs  # Show futures agent logs
```

## 📊 POSITION ANALYSIS

### Intelligent Position Managers
```bash
# Crypto positions with intelligent TP/SL analysis
python3 manual_position_manager.py status

# Futures positions with intelligent TP/SL analysis  
python3 intelligent_futures_manager.py status

# Monitor positions in real-time
python3 manual_position_manager.py monitor          # Crypto (60s intervals)
python3 intelligent_futures_manager.py monitor      # Futures (30s intervals)
```

### Manual Position Management
```bash
# Close profitable positions
python3 manual_position_manager.py close_tp

# Close losing positions
python3 manual_position_manager.py close_sl

# Close specific position
python3 manual_position_manager.py close BTCUSD

# Close all positions
python3 manual_position_manager.py close ALL

# Same commands for futures
python3 intelligent_futures_manager.py close BTCUSD
```

## ⚙️ CONFIGURATION

### Environment Variables
```bash
# Intelligent TP/SL Controls
export TB_INTELLIGENT_CRYPTO_TPSL=1    # Enable for hybrid trader (default: ON)
export TB_INTELLIGENT_FUTURES_TPSL=1   # Enable for futures agent (default: ON)

# Trading Controls
export TB_NO_TRADE=1                   # Dry run mode
export TB_OFFLINE=1                    # Offline mode (synthetic data)

# Risk Management
export TB_USE_ENHANCED_RISK=1          # Enhanced risk management
export TB_USE_ADAPTIVE_STRATEGY=1      # Adaptive strategy
export TB_USE_REGIME_DETECTION=1       # Market regime detection

# ML Controls
export TB_USE_ML_GATE=1                # ML gating
export TB_ML_GATE_MIN_PROB=0.6         # ML confidence threshold

# Notifications
export TB_TRADER_NOTIFY=1              # Enable notifications
export TB_ENABLE_DISCORD=1             # Discord notifications
export TB_NO_TELEGRAM=0                # Telegram notifications
```

### Config Files
```bash
# Main configuration
config/promoted_params.json              # TP/SL fallback values, risk settings

# Futures configuration  
futures_agent_config.env                 # Futures-specific settings

# Environment configuration
.env                                      # API keys, webhooks, etc.
```

## 🧪 TESTING & DEBUGGING

### Dry Run Testing
```bash
# Test hybrid trader (single run)
export TB_NO_TRADE=1 && python3 scripts/hybrid_crypto_trader.py

# Test futures agent (single run)
export TB_NO_TRADE=1 && python3 high_risk_futures_agent.py

# Test with offline data
export TB_OFFLINE=1 TB_NO_TRADE=1 && python3 scripts/hybrid_crypto_trader.py
```

### Live Testing
```bash
# Start in paper trading mode
export TB_NO_TRADE=0 && ./dual_agent.sh start

# Monitor in real-time
watch -n 10 "./dual_agent.sh status"

# Check recent activity
tail -f main_agent.log
tail -f futures_agent.log
```

### Debug Tools
```bash
# Check API connectivity
python3 debug_sources.py

# Test Perplexity auth
python3 test_pplx_auth.py

# Check environment
python3 debug_env.py

# Health check
bash scripts/health_check.sh
```

## 📈 BACKTESTING & OPTIMIZATION

### Backtesting
```bash
# Run backtest with current parameters
python3 scripts/run_backtest.py

# Parameter optimization
python3 scripts/weekly_propose_canary.sh
```

### Performance Analysis
```bash
# Asset hit rate analysis
python3 scripts/asset_hit_rate.py

# Evaluation runner
python3 scripts/eval_runner.py

# ML retraining
python3 scripts/ml_retrainer.py --bars_dir bars --out_root eval_runs/ml
```

## 🔍 LOG ANALYSIS

### Real-time Monitoring
```bash
# Follow main logs
tail -f trader_loop.log                  # Hybrid trader loop
tail -f high_risk_futures_loop.log       # Futures loop
tail -f position_manager.log             # Position monitoring

# Follow specific logs
tail -f main_agent.log                   # Main agent activity  
tail -f futures_agent.log                # Futures agent activity
tail -f trading_agent.log                # General trading activity
```

### Historical Analysis
```bash
# Check recent decisions
grep "SIGNAL\|EXIT\|BUY\|SELL" main_agent.log | tail -20

# Check TP/SL hits
grep "profit\|stop\|loss" main_agent.log | tail -10

# Check intelligent targets
grep "intelligent\|quality" main_agent.log | tail -10
```

## 🛠️ MAINTENANCE

### Cleanup
```bash
# Clean old logs (keep last 7 days)
find . -name "*.log" -mtime +7 -delete

# Clean old artifacts
find runs/ -name "*.json" -mtime +30 -delete
find bars/ -name "*.csv" -mtime +30 -delete
```

### Updates
```bash
# Update promoted parameters
python3 scripts/weekly_propose_canary.sh

# Retrain ML models
python3 scripts/ml_retrainer.py --bars_dir bars --out_root eval_runs/ml --link_dir eval_runs/ml/latest
```

## 🚨 EMERGENCY CONTROLS

### Emergency Stop
```bash
# Stop everything immediately
./dual_agent.sh stop

# Kill all trading processes
pkill -f "hybrid_crypto_trader.py"
pkill -f "high_risk_futures_agent.py"

# Close all positions manually
python3 manual_position_manager.py close ALL
python3 intelligent_futures_manager.py close ALL
```

### Recovery
```bash
# Restart after emergency stop
./dual_agent.sh restart

# Check system health
bash scripts/health_check.sh

# Verify positions
python3 manual_position_manager.py status
python3 intelligent_futures_manager.py status
```

## 📊 INTELLIGENT TP/SL SYSTEM REFERENCE

### Crypto Targets (Hybrid Trader)
- 🔥 **Excellent**: 12-20% TP, 5% SL (High conviction, strong confluence)
- ✅ **Good**: 8-12% TP, 4% SL (Good setup, decent confidence)
- ⚡ **Fair**: 5-8% TP, 3% SL (Marginal setup, lower confidence)

### Futures Targets (Futures Agent)  
- 🔥 **Excellent**: 15-25% TP, 6% SL (High conviction + leverage)
- ✅ **Good**: 10-15% TP, 5% SL (Good setup + leverage)
- ⚡ **Fair**: 6-10% TP, 4% SL (Marginal setup + leverage)

### Asset Difficulty Multipliers
- **BTC**: 1.5x (Hardest to move)
- **ETH**: 1.3x (Major crypto)
- **SOL**: 1.1x (Top 10)
- **Others**: 0.7x-1.0x (Easier alts)

## 📱 QUICK STATUS CHECKS

### One-liner Status
```bash
# Everything in one command
./dual_agent.sh status && echo "=== POSITIONS ===" && python3 manual_position_manager.py status | head -20
```

### Performance Summary
```bash
# Portfolio value + recent activity
grep -E "(Portfolio Value|P&L)" <(python3 manual_position_manager.py status) && tail -5 main_agent.log
```
