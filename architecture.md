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
- `high_risk_futures_agent.py` — Futures trading orchestrator
- `futures_integration.py` — Futures platform integration layer
- `futures_trading_platform.py` — Multi-platform futures trading support
- `futures_paper_trading_demo.py` — Paper trading demonstrations
- `dual_agent.sh` — Dual-agent management script

### Data Flow (Enhanced)
1) **Data Ingestion**: Multi-source data fetch (Alpaca, Yahoo, Binance, CoinGecko, Perplexity)
2) **Signal Processing**: ML gates, ATR filters, adaptive thresholds, market regime detection
3) **Risk Management**: Kelly sizing, portfolio VaR, correlation controls, position limits
4) **Execution**: Independent execution for spot (Alpaca) and futures (Binance/Bybit)
5) **Monitoring**: Real-time notifications, heartbeat monitoring, performance tracking
6) **Persistence**: SQLite, JSON artifacts, CSV data, auto-commit to git

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
