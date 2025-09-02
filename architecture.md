# Project Tracer Bullet — Architecture

## Current Architecture (V4.3 - September 2, 2025)

### Dual-Agent System
- **Main Agent (Low-Risk)**: Enhanced hybrid crypto trader with ML gates and adaptive strategies
- **Futures Agent (High-Risk)**: Leveraged futures/perpetuals agent with momentum trading
- **Shared Infrastructure**: Common data sources, monitoring, and notification systems
- **Independent Operation**: Separate capital allocation, risk management, and execution

### Main Agent Components
- `scripts/hybrid_crypto_trader.py` — Main orchestrator with ML gates and adaptive thresholds
- `config.py` — Environment configuration and settings management
- `alpaca.py` — Multi-source data fetch (Alpaca, Yahoo, Binance, CoinGecko)
- `multi_source_data_provider.py` — Unified data provider with failover
- `sentiment_utils.py` — Robust sentiment analysis with outlier removal
- `divergence.py` — Divergence computation with adaptive thresholds
- `enhanced_risk_manager.py` — Advanced risk management with Kelly sizing
- `market_regime_detector.py` — Real-time market regime classification
- `adaptive_strategy.py` — Parameter optimization and exploration
- `autocommit_enhanced.py` — Enhanced auto-commit system for artifacts

### Futures Agent Components
- `high_risk_futures_agent.py` — Futures trading orchestrator with UTC timestamps
- `futures_integration.py` — Futures platform integration layer with account balance retrieval
- `futures_trading_platform.py` — Multi-platform futures trading support with real API calls
- `futures_paper_trading_demo.py` — Paper trading demonstrations
- `dual_agent.sh` — Dual-agent management script

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
