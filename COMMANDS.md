# 🚀 Production-Ready 24/7 Trading System - Complete Command Reference
# =====================================================================

## 🎯 PRODUCTION DEPLOYMENT READY (September 4, 2025)

**SYSTEM STATUS:** ✅ 95% confidence level achieved for 24/7 autonomous deployment
**INFRASTRUCTURE:** Complete monitoring, enhanced signal intelligence, unified trade management
**DEPLOYMENT:** Single-command startup with comprehensive initialization

---

## 🚀 UNIFIED DEPLOYMENT SYSTEM

### Single-Command Production Startup
```bash
# Deploy complete trading system (recommended)
./start_trading_system.sh

# System includes:
# - Phase 1C ultra-aggressive thresholds
# - Enhanced signal intelligence (0-10 scale)
# - Confidence-based position sizing
# - Complete monitoring and auto-recovery
# - Database logging and audit trails
# - Enhanced notifications (Discord + Telegram)
```

### Legacy Dual Agent Control (Alternative)
```bash
# Start both enhanced agents
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

---

## 📊 ENHANCED SYSTEM MONITORING

### Production Health Checks
```bash
# Comprehensive system status
bash scripts/health_check.sh

# Real-time monitoring dashboard  
bash scripts/monitoring_status.sh

# Check watchdog status
ps aux | grep watchdog

# Check cron jobs
crontab -l | grep tracer
```

### Enhanced Signal Intelligence Monitoring
```bash
# View signal quality scores in logs
grep "signal_quality" main_agent.log | tail -20

# Monitor regime detection
grep "market_regime" main_agent.log | tail -10

# Check conviction scoring
grep "conviction_score" main_agent.log | tail -15

# Enhanced notification status
grep "enhanced_notification" main_agent.log | tail -10
```

---

## 🎛️ ENHANCED CONFIGURATION

### Phase 1C Ultra-Aggressive Settings
```bash
# Phase 1C Configuration (already in .env)
export TB_MIN_SIGNAL_QUALITY=1.0       # Ultra-aggressive quality threshold
export TB_MIN_CONVICTION_SCORE=2.0     # Low conviction requirement
export TB_USE_ENHANCED_SIGNALS=1       # Enable enhanced signal system
export TB_ENHANCED_NOTIFICATIONS=1     # Rich Discord/Telegram notifications

# Enhanced Trade Management
export TB_ENHANCED_TRADE_MANAGER=1     # Confidence-based sizing
export TB_DAILY_TRADE_LIMIT=8          # Max trades per agent per day
export TB_CONFIDENCE_BASED_SIZING=1    # Dynamic position sizing (20%-100%)

# Monitoring & Recovery
export TB_ENABLE_WATCHDOG=1            # Auto-restart on failure
export TB_HEALTH_CHECK_INTERVAL=900    # Health checks every 15 minutes
export TB_AUTO_RECOVERY=1              # Self-healing mechanisms
```

### Enhanced Signal Intelligence Controls
```bash
# Signal Quality Scoring (0-10 scale)
export TB_MIN_SIGNAL_QUALITY=1.0       # Quality threshold (ultra-aggressive)
export TB_SENTIMENT_WEIGHT=0.4         # Sentiment factor weight
export TB_MOMENTUM_WEIGHT=0.3          # Price momentum weight  
export TB_VOLUME_WEIGHT=0.2            # Volume confirmation weight
export TB_RSI_WEIGHT=0.1               # RSI extreme bonus weight

# Market Regime Detection
export TB_REGIME_DETECTION=1           # Enable regime classification
export TB_VOLATILITY_REGIMES=1         # Low/normal/high/extreme volatility
export TB_TREND_REGIMES=1              # Bull/bear/sideways trend detection
export TB_REGIME_CONFIDENCE=1          # Regime confidence scoring

# Conviction Scoring System
export TB_MIN_CONVICTION_SCORE=2.0     # Conviction threshold (ultra-aggressive)
export TB_QUALITY_WEIGHT=0.4           # Signal quality factor
export TB_REGIME_WEIGHT=0.3            # Regime alignment factor
export TB_VOLATILITY_WEIGHT=0.2        # Volatility assessment factor
export TB_CONFIRMATION_WEIGHT=0.1      # Confirmation signals factor
```
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
