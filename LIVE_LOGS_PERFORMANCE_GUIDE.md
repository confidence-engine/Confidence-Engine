# 📊 Live Logging Locations for Performance Assessment

*Generated: September 4, 2025*

Based on the current running processes and log analysis, here are the **exact live log locations** for both agents:

## 🔥 Futures Agent Live Logs

### Primary Live Trading Log
- **File**: `market_regime_detector.log` 
- **Size**: **4.2MB** (47,094 lines)
- **Content**: Real-time futures positions, P&L, leveraged ROI calculations
- **Sample Recent Activity**:
  ```
  ⚡ DOGEUSDT: Leveraged ROI=-7.4% (Price: -0.30% × 25x leverage)
  ⚡ ALGOUSDT: Leveraged ROI=+14.3% (Price: +0.57% × 25x leverage) 
  ✅ DOGEUSDT position OK, no exit condition met
  ```

### Process Management Log
- **File**: `high_risk_futures_loop.log`
- **Size**: 1.3KB (48 lines)
- **Content**: Loop restarts, crash recovery, process health
- **Current Status**: Running with PID 99296 (30-second intervals)

### Simple Startup Log
- **File**: `futures_agent_simple.log`
- **Size**: 0 bytes (not currently used)

## 💎 Hybrid Agent Live Logs

### Primary Trading Decisions Log
- **File**: `trading_agent.log`
- **Size**: **313KB** (3,241 lines)
- **Content**: Multi-asset trading decisions, signal analysis, position management
- **Auto-Commit**: ✅ Every 1-2 minutes

### Loop Management Log
- **File**: `trader_loop.log`
- **Size**: 935 bytes (20 lines)
- **Content**: ML gate parameters, exploration modes, epsilon-greedy settings
- **Recent Activity**:
  ```
  [start_hybrid_loop] gate PROB=0.25 ATR=0.002 mode=normal
  [start_hybrid_loop] gate PROB=0.25 ATR=0.0005 mode=epsilon
  ```

### Simple Startup Log
- **File**: `hybrid_agent_simple.log` 
- **Size**: 0 bytes (not currently used)

## 📈 Shared Performance Databases

| **Database** | **Agent Coverage** | **Key Metrics** |
|--------------|-------------------|-----------------|
| `enhanced_trading.db` | Both agents | ML features, enhanced metrics |
| `validation_analysis.db` | Both agents | Backtesting, model validation |
| `tracer.db` | Both agents | Core trading state |

## 🎯 Real-Time Position Tracking

### Futures Agent State
- **Location**: `market_regime_detector.log` (live P&L tracking)
- **Current Positions**: DOGEUSDT (long, -7.4% ROI), ALGOUSDT (long, +14.3% ROI)
- **Update Frequency**: Every 30 seconds

### Hybrid Agent State
- **Location**: `state/hybrid_trader_state_{SYMBOL}.json`
- **Current Status**: All positions = 0 (no active positions)
- **Update Frequency**: Real-time on position changes

## ⚡ Monitoring Commands for Future Analysis

```bash
# Real-time futures performance
tail -f market_regime_detector.log | grep -E "(Leveraged ROI|position OK|exit condition)"

# Hybrid agent decisions
tail -f trading_agent.log | grep -E "(TRADE|POSITION|SIGNAL)"

# Overall system health
tail -f trader_loop.log high_risk_futures_loop.log

# Position status check
cat state/*.json | jq '.position_qty // empty' | grep -v null
```

## 📊 Complete Performance Data Infrastructure

### Primary Log Files (Auto-Committed to Git)

| **Log File** | **Size** | **Lines** | **Purpose** | **Auto-Commit Status** |
|--------------|----------|-----------|-------------|------------------------|
| `market_regime_detector.log` | **4.2MB** | **47,094** | Market analysis & regime detection | ✅ **Active** |
| `trading_agent.log` | **313KB** | **3,241** | Main trading decisions & execution | ✅ **Active** |
| `trader_loop.log` | **935B** | **20** | Loop status & health monitoring | ✅ **Active** |
| `high_risk_futures_loop.log` | **1.3KB** | **48** | Futures agent execution log | ✅ **Active** |
| `health_check_cron.log` | **1.3KB** | **24** | System health monitoring | ✅ **Active** |

### SQLite Databases (Critical Performance Data)

| **Database** | **Size** | **Purpose** | **Auto-Commit Status** |
|--------------|----------|-------------|------------------------|
| `enhanced_trading.db` | **60KB** | Enhanced trading metrics & ML features | ✅ **Active** |
| `validation_analysis.db` | **592KB** | Backtesting results & model validation | ✅ **Active** |
| `tracer.db` | **148KB** | Core trading state & historical data | ✅ **Active** |

### Structured Data Directories

#### 📁 `runs/` Directory (1,921 Files)
- **Purpose**: JSON artifacts from each trading execution
- **Contents**: Signal analysis, divergence computation, trade decisions
- **Sample Files**: `10.json`, `11.json`, `15_accepted.txt`
- **Auto-Commit**: ✅ **Every 1-2 minutes**

#### 📁 `state/` Directory (Real-Time Trading State)
- **Purpose**: Current position tracking & trade journal
- **Key Files**:
  - `global_state.json`: System heartbeat & last run timestamps
  - `hybrid_trader_state_{SYMBOL}.json`: Per-asset position tracking
  - `crypto_trader_state.json`: Legacy trader state
- **Auto-Commit**: ✅ **Immediate on changes**

#### 📁 `bars/` Directory (37 Files)
- **Purpose**: Historical price data for backtesting
- **Contents**: CSV files with OHLCV data per trading run
- **Auto-Commit**: ✅ **Active**

#### 📁 `eval_runs/` Directory (34 Subdirectories)
- **Purpose**: ML model evaluation & parameter optimization
- **Recent Activity**: Live auto-apply configurations every 2 minutes
- **Auto-Commit**: ✅ **Active**

## 🔧 Auto-Commit Configuration

```bash
# Active Environment Variables
TB_AUTOCOMMIT_ARTIFACTS=1  # Commits all trading artifacts
TB_AUTOCOMMIT_PUSH=1       # Pushes to remote repository
```

**Auto-Commit Policy**:
- ✅ **Commits**: All logs, JSON artifacts, CSV data, databases
- ❌ **Blocks**: Python code (.py), shell scripts (.sh), environment files (.env)
- ⏱️ **Frequency**: Every 1-2 minutes during active trading
- 🔄 **Push**: Automatic push to remote git repository

## 📈 Performance Analysis Readiness

### Short-Term Performance (1-7 days)
- **Source**: `trading_agent.log` + `state/*.json`
- **Metrics**: Trade execution, P&L, position tracking
- **Frequency**: Real-time updates

### Medium-Term Analysis (1-4 weeks)
- **Source**: `market_regime_detector.log` + `enhanced_trading.db`
- **Metrics**: Signal quality, regime detection accuracy, ML features
- **Frequency**: Continuous data collection

### Long-Term Optimization (1+ months)
- **Source**: `validation_analysis.db` + `eval_runs/`
- **Metrics**: Backtesting results, parameter optimization, model performance
- **Frequency**: Weekly parameter updates

## 📋 Data Retention & Storage

| **Data Type** | **Retention** | **Storage Location** | **Analysis Ready** |
|---------------|---------------|---------------------|-------------------|
| **Live Trading** | Permanent | `state/` + `trading_agent.log` | ✅ **Ready** |
| **Signal Analysis** | Permanent | `runs/` + `market_regime_detector.log` | ✅ **Ready** |
| **Price Data** | Permanent | `bars/` + databases | ✅ **Ready** |
| **ML Metrics** | Permanent | `eval_runs/` + `validation_analysis.db` | ✅ **Ready** |
| **System Health** | 30 days | Various `.log` files | ✅ **Ready** |

## 🎯 Future Performance Analysis Workflow

1. **Daily P&L Tracking**: Query `state/` JSON files + `trading_agent.log`
2. **Signal Quality Analysis**: Parse `market_regime_detector.log` (47K+ lines of data)
3. **ML Model Performance**: Query `validation_analysis.db` + `eval_runs/`
4. **System Reliability**: Monitor `health_check_cron.log` + `trader_loop.log`
5. **Comparative Analysis**: Cross-reference all data sources for comprehensive insights

## 🎯 Key Insight for Performance Assessment

**Primary Data Sources**:
1. **Futures Performance**: `market_regime_detector.log` (4.2MB of real trading data)
2. **Hybrid Performance**: `trading_agent.log` + `state/*.json` files
3. **ML Metrics**: `validation_analysis.db` + `eval_runs/` directories
4. **System Health**: `trader_loop.log` + `high_risk_futures_loop.log`

All logs are **auto-committed to git** every 1-2 minutes, ensuring complete historical performance tracking for future analysis and optimization workflows.

---

*This document provides a comprehensive overview of all performance data storage locations across the trading system for enabling systematic trading system optimization.*
