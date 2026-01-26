# 🚀 Trading Agents Startup Guide - Phase 1C Deployment

**Status**: ❌ **Agents NOT Currently Running**  
**Date**: September 3, 2025  
**Configuration**: Phase 1C Ultra-Aggressive Thresholds Applied

## 📊 Current Status

### Agents Status: ❌ OFFLINE
```bash
Process Status: No trading agents currently running
Last Activity: 
- Hybrid Agent: Last seen Sep 3 06:56 (trader_loop.err)
- Futures Agent: Last seen Sep 3 04:52 (high_risk_futures_loop.err)
```

### Phase 1C Configuration: ✅ READY
```bash
TB_MIN_SIGNAL_QUALITY=1.0      # Ultra-permissive
TB_MIN_CONVICTION_SCORE=2.0    # Ultra-permissive  
TB_USE_ENHANCED_SIGNALS=1      # Enhanced signals active
TB_TRADER_OFFLINE=1            # Paper trading mode
TB_NO_TRADE=1                  # Simulation mode
```

## 🚀 Commands to Start Both Agents

### 1. Start Hybrid Crypto Agent
```bash
cd /Users/mouryadamarasing/Documents/Project-Tracer-Bullet

# Option A: Direct start with Phase 1C config
export TB_MIN_SIGNAL_QUALITY=1.0
export TB_MIN_CONVICTION_SCORE=2.0
export TB_USE_ENHANCED_SIGNALS=1
export TB_TRADER_OFFLINE=1
export TB_NO_TRADE=1
bash scripts/start_hybrid_loop.sh

# Option B: Start with existing script (contains auto-config)
bash scripts/start_hybrid_loop.sh
```

### 2. Start Futures Agent
```bash
cd /Users/mouryadamarasing/Documents/Project-Tracer-Bullet

# Option A: Direct start with Phase 1C config  
export TB_MIN_SIGNAL_QUALITY=1.0
export TB_MIN_CONVICTION_SCORE=2.0
export TB_USE_ENHANCED_SIGNALS=1
bash scripts/start_futures_loop.sh

# Option B: Start with existing script
bash scripts/start_futures_loop.sh
```

### 3. Start Both Agents (Recommended)
```bash
cd /Users/mouryadamarasing/Documents/Project-Tracer-Bullet

# Set Phase 1C environment
export TB_MIN_SIGNAL_QUALITY=1.0
export TB_MIN_CONVICTION_SCORE=2.0
export TB_USE_ENHANCED_SIGNALS=1
export TB_TRADER_OFFLINE=1
export TB_NO_TRADE=1

# Start both agents
echo "🚀 Starting both trading agents with Phase 1C configuration..."
bash scripts/start_hybrid_loop.sh
sleep 5
bash scripts/start_futures_loop.sh

echo "✅ Both agents starting in background"
echo "📊 Check status with: ps aux | grep -E '(hybrid_crypto_trader|high_risk_futures_agent)'"
```

## 🛡️ Watchdog & Retry Mechanisms

### ✅ Built-in Retry Mechanisms

**Hybrid Agent (`start_hybrid_loop.sh`):**
- ✅ Infinite loop with 60-second intervals
- ✅ Auto-restart on crashes (`|| true`)
- ✅ Exploration windows and epsilon-greedy
- ✅ Parameter auto-adjustment
- ✅ ML retraining loop
- ✅ Error logging to `trader_loop.err`

**Futures Agent (`start_futures_loop.sh`):**
- ✅ Infinite loop with crash recovery
- ✅ 30-second trading intervals for real-time risk
- ✅ 60-second restart delay on crashes
- ✅ Internet connectivity checks in agent code
- ✅ Error logging to `high_risk_futures_loop.err`

**Internet Outage Handling:**
- ✅ **Futures Agent**: Has `check_internet_connectivity()` method
- ✅ **Multiple endpoint testing**: httpbin.org, binance.com, google.com
- ✅ **Fallback methods**: HTTP requests + socket tests + DNS resolution
- ✅ **Graceful degradation**: Continues with limited functionality
- ✅ **Max failure tracking**: Enters offline mode after 5 consecutive failures

### ✅ Health Check System

**Daily Health Monitoring (`scripts/health_check.sh`):**
- ✅ Process running verification
- ✅ Log freshness checks (30-minute max age)
- ✅ Recent artifacts validation
- ✅ ML model health monitoring
- ✅ Discord/Telegram alerts on failures
- ✅ Self-healing mechanisms

**Cron Setup:**
```bash
# Add to crontab for automated health checks
0 */6 * * * cd /Users/mouryadamarasing/Documents/Project-Tracer-Bullet && bash scripts/health_check.sh
```

### ✅ Watchdog Services (Optional)

**Futures Watchdog:**
```bash
# Setup automated futures agent monitoring
bash setup_futures_watchdog.sh

# Manual watchdog script
bash scripts/watchdog_futures.sh
```

**Hybrid Watchdog:**
```bash
# Manual watchdog script  
bash scripts/watchdog_hybrid.sh
```

## 📱 Monitoring Commands

### Check if Agents are Running
```bash
ps aux | grep -E "(hybrid_crypto_trader|high_risk_futures_agent)" | grep -v grep
```

### Monitor Logs in Real-time
```bash
# Hybrid agent
tail -f trader_loop.log

# Futures agent  
tail -f high_risk_futures_loop.log

# Both agents
tail -f trader_loop.log high_risk_futures_loop.log
```

### Check Recent Activity
```bash
# Last few log entries
tail -10 trader_loop.err
tail -10 high_risk_futures_loop.err

# Health check status
bash scripts/health_check.sh
```

### Stop Agents
```bash
# Kill hybrid agent
pkill -f "hybrid_crypto_trader.py"

# Kill futures agent
pkill -f "high_risk_futures_agent.py" 

# Kill both
pkill -f "(hybrid_crypto_trader|high_risk_futures_agent).py"
```

## 🎯 Expected Behavior with Phase 1C

### When Working Correctly:
- **Signal Generation**: Multiple signals per cycle (quality 1-4/10 range)
- **Trade Frequency**: 3-5 trades per week target
- **Log Activity**: New entries every 30-60 seconds
- **Discord Notifications**: Trade alerts with signal quality metrics
- **Error Handling**: Graceful failures, auto-restart

### Alert Conditions:
- Process stops unexpectedly → Auto-restart
- Internet outage → Offline mode, retry mechanism
- API failures → Circuit breaker, fallback data
- Log staleness → Health check alert
- No signals → Threshold adjustment

## 🚨 Troubleshooting

### Common Issues:

**1. Import Errors**
- Fixed in futures agent ✅
- Hybrid agent needs class name fix ⚠️

**2. API Key Issues** 
- Perplexity keys invalid (expected in offline mode)
- Alpaca keys missing (expected in paper mode)

**3. Precision Errors**
- Some futures symbols have precision issues
- Fallback mechanisms in place

**4. No Signals Generated**
- Phase 1C should fix this ✅
- Ultra-aggressive thresholds (1.0/2.0) implemented

## 🎉 Quick Start (TL;DR)

```bash
cd /Users/mouryadamarasing/Documents/Project-Tracer-Bullet

# Phase 1C ultra-aggressive configuration
export TB_MIN_SIGNAL_QUALITY=1.0
export TB_MIN_CONVICTION_SCORE=2.0
export TB_USE_ENHANCED_SIGNALS=1
export TB_TRADER_OFFLINE=1
export TB_NO_TRADE=1

# Start both agents
bash scripts/start_hybrid_loop.sh
bash scripts/start_futures_loop.sh

# Monitor
tail -f trader_loop.log high_risk_futures_loop.log
```

**Result**: Both agents running with enhanced signal intelligence and ultra-aggressive thresholds for active trading! 🚀
