# Project Tracer Bullet — Architecture

## Current Architecture (V5.0 - September 4, 2025)

### Production-Ready Dual-Agent System
- **Main Agent (Hybrid Crypto)**: Enhanced signal intelligence with Phase 1C ultra-aggressive configuration
- **Futures Agent (High-Risk)**: Leveraged futures/perpetuals with enhanced signal parity
- **Unified Infrastructure**: Industrial-grade monitoring, trade management, and notification systems
- **Autonomous Operation**: 24/7 capability with comprehensive monitoring and auto-recovery

### Enhanced Signal Intelligence System (V5.0 Breakthrough)
- **Signal Quality Scoring**: 0-10 scale assessment replacing ML complexity
- **Market Regime Detection**: Multi-dimensional volatility/trend/volume classification
- **Conviction Scoring**: Weighted factor combination for holistic trade assessment
- **Regime-Aware Trading**: Adaptive logic with different thresholds per market condition
- **Unified Implementation**: Same enhanced system across both hybrid and futures agents

### Core System Components

#### Enhanced Infrastructure (Production-Ready)
- `start_trading_system.sh` — Unified deployment script with comprehensive initialization
- `enhanced_trade_manager.py` — Intelligent trade management with confidence-based sizing
- `enhanced_notifications.py` — Unified Discord/Telegram integration with delivery tracking
- `enhanced_db_logger.py` — Comprehensive audit trails and database logging
- `enhanced_trading.db` — SQLite database with 6 tables for complete trade tracking

#### Main Agent Components (Enhanced Signal Intelligence)
- `scripts/hybrid_crypto_trader.py` — Enhanced orchestrator with signal quality system (143K+ bytes)
- `evaluate_enhanced_signals()` — Core enhanced signal evaluation function
- `market_regime_detector.py` — Real-time regime classification with confidence scoring
- `advanced_risk_manager.py` — Kelly criterion and portfolio VaR management
- `adaptive_strategy.py` — Parameter optimization with exploration windows

#### Futures Agent Components (Signal Parity)
- `high_risk_futures_agent.py` — Enhanced futures orchestrator with signal parity (92K+ bytes)
- `evaluate_enhanced_futures_signals()` — Futures-optimized enhanced signal evaluation
- `futures_trading_platform.py` — Real API integration with dynamic precision handling
- `intelligent_futures_manager.py` — Advanced position management with leverage controls

#### Monitoring & Recovery Infrastructure
- `scripts/watchdog_hybrid.sh` — Main agent monitoring with process checks and log validation
- `scripts/watchdog_futures.sh` — Futures agent monitoring every 60 seconds with auto-restart
- `scripts/health_check.sh` — Comprehensive system health validation (7,856 bytes)
- **Launchd Services**: macOS integration for persistent background monitoring
- **Cron Jobs**: Automated health checks every 15 minutes with self-healing

### Data Flow (Enhanced V5.0)
1) **Enhanced Data Ingestion**: Multi-source with Perplexity Pro API key rotation
2) **Signal Quality Assessment**: 0-10 scale scoring based on sentiment, momentum, volume, RSI
3) **Market Regime Detection**: Real-time classification with confidence levels
4) **Conviction Scoring**: Weighted combination of quality, regime, volatility, confirmation
5) **Intelligent Trade Management**: Confidence-based position sizing (20%-100% of max)
6) **Enhanced Execution**: Independent execution with dynamic precision and real API integration
7) **Comprehensive Monitoring**: Multi-layer monitoring with automatic recovery
8) **Advanced Persistence**: SQLite database + JSON artifacts + auto-commit system

### Phase 1C Configuration (Ultra-Aggressive)
- **TB_MIN_SIGNAL_QUALITY**: 1.0 (extremely permissive for maximum activity)
- **TB_MIN_CONVICTION_SCORE**: 2.0 (low threshold for trade activation)
- **Daily Trade Limits**: 8 trades per agent maximum
- **Position Sizing**: Confidence-based dynamic sizing (20%-100% of maximum)
- **Enhanced Notifications**: Rich Discord/Telegram with emoji indicators and metrics

### Data Flow (Enhanced)
1) **Data Ingestion**: Multi-source data fetch (Alpaca, Yahoo, Binance, CoinGecko, Perplexity)
2) **Signal Processing**: ML gates, ATR filters, adaptive thresholds, market regime detection
3) **Risk Management**: Kelly sizing, portfolio VaR, correlation controls, position limits
4) **Execution**: Independent execution for spot (Alpaca) and futures (Binance/Bybit)
5) **Monitoring**: Real-time notifications, heartbeat monitoring, performance tracking
6) **Persistence**: SQLite, JSON artifacts, CSV data, auto-commit to git

### Recent Enhancements (September 2, 2025)
- **Real API Integration**: Fixed `get_positions()` to return actual Binance testnet data instead of empty lists
- **Account Balance**: Added `get_account_balance()` function for real-time balance monitoring
- **UTC Timestamps**: Standardized all timestamps to `datetime.now(timezone.utc).isoformat()`
- **Position Tracking**: Now properly monitors 8 active positions with real P&L data
- **API Authentication**: HMAC SHA256 signature generation for secure API calls
- **Error Handling**: Graceful degradation when API unavailable with proper logging

## Data Flow
1) Fetch bars/headlines (`alpaca.py`)
2) Filter and summarize headlines (`narrative_dev.py` with relevance)
3) Score sentiment (`finbert.py`, relevant-only)
4) Compute price score and volume z (`price.py`)
5) Blend and decay narrative (`narrative_analysis.py`)
6) Compute divergence and reason (`divergence.py`)
7) Explain labels and summaries (`explain.py`)
8) Persist run (`db.py`; runs table)
9) Export artifacts (`export.py`): JSON to `runs/<id>.json`, bars to `bars/<id>.csv`
10) Enrichment (universe runs): persist `evidence_line`, `thesis {action,risk_band,readiness}`, and per‑TF `plan` snapshots back into saved `universe_runs/*.json`
11) Optional notifications: Telegram/Discord (env‑gated)
12) Optional git ops: when enabled, artifacts and enrichment deltas are auto‑staged/committed (and pushed when allowed)

## Persistence
- SQLite: `tracer.db` (runs table)
- JSON: `runs/<run_id>.json` (single‑asset), `universe_runs/*.json` (universe scans)
- CSV: `bars/<run_id>.csv`, `universe_runs/metrics.csv`
- Git ops (env‑gated): `TB_UNIVERSE_GIT_AUTOCOMMIT`, `TB_UNIVERSE_GIT_PUSH` for universe; similar flags for evaluation artifacts

## Environments
- .env controls keys and thresholds; no quotes for values.

## 🛡️ Comprehensive Monitoring System (V4.4 - Latest Addition)

### System Architecture
- **Watchdog Layer**: Independent monitoring processes that detect and restart failed agents
- **Health Check Layer**: Comprehensive system status verification with self-healing capabilities
- **Notification Layer**: Multi-channel alerts (Discord, Telegram) for critical events
- **Recovery Layer**: Automatic restart mechanisms with graceful degradation

### Core Components
- **Futures Watchdog** (`scripts/watchdog_futures.sh`): Monitors futures agent every 60 seconds
- **Hybrid Watchdog** (`scripts/watchdog_hybrid.sh`): Monitors main crypto trader with process checks
- **Health Check Script** (`scripts/health_check.sh`): Comprehensive system status verification
- **Monitoring Status** (`scripts/monitoring_status.sh`): Real-time dashboard for both agents
- **Launchd Services**: macOS launchd integration for persistent background monitoring

### Monitoring Features
- **Process Monitoring**: Checks for running processes and automatic restart on failure
- **Log Freshness**: Validates recent activity in log files
- **ML Model Health**: Monitors model loading, inference, and performance metrics
- **Database Integrity**: Verifies SQLite connectivity and recent data
- **API Connectivity**: Tests external API endpoints and authentication
- **Resource Usage**: Monitors system resources and memory usage

### Recovery Mechanisms
- **Automatic Restart**: Failed agents are automatically restarted with retry logic
- **Graceful Degradation**: Fallback modes when components fail (ML gate bypass, reduced features)
- **Circuit Breakers**: Prevents cascading failures with timeout and retry limits
- **Self-Healing**: Health checks can trigger recovery actions automatically

### Configuration Management
- **Environment Variables**: All monitoring features controlled via .env flags
- **Flexible Intervals**: Configurable monitoring frequencies and thresholds
- **Multi-Level Alerts**: Different alert levels for different types of issues
- **Audit Logging**: Comprehensive logging of all monitoring events and actions

### Integration Points
- **Git Auto-Commit**: Monitoring artifacts are automatically committed to repository
- **Notification Channels**: Discord and Telegram integration for real-time alerts
- **Database Logging**: All monitoring events logged to SQLite for analysis
- **External APIs**: Integration with external monitoring services if needed

### Security Considerations
- **Credential Protection**: API keys and sensitive data properly secured
- **Access Control**: Monitoring scripts run with appropriate permissions
- **Audit Trail**: All monitoring actions are logged for security review
- **Fail-Safe Design**: Monitoring system itself has redundancy and recovery mechanisms
