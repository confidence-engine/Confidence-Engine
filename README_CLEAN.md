# Confidence Engine — Dual-Agent Trading System

A pragmatic research/trading agent that exploits the gap between narrative (news) and tape (price) with discipline, explainability, and continuous evaluation.

**Current Status (September 2, 2025)**: Dual-agent system operational with both low-risk and high-risk trading agents running simultaneously.

---

## 1) What it does
- **Dual-Agent Architecture**: Independent main agent (low-risk spot trading) and futures agent (high-risk leveraged trading)
- Detects divergences between structured news momentum and multi‑timescale price action
- Produces rich per‑asset artifacts with narratives, evidence lines, confidence, and Polymarket mapping
- Delivers optional digest messages to Telegram and Discord (env‑gated, safe split/chunking)
- Continuously evaluates probability quality (Brier, log‑loss, calibration) and auto‑publishes CSVs
- **Enhanced Features**: ML gates with 37 technical indicators, market regime detection, correlation filtering, smart leverage

---

## 2) Current Live Configuration
- **Main Agent**: 18 crypto assets with ML gates, ATR filters, and adaptive thresholds
- **Futures Agent**: 10 crypto futures with momentum trading and 25x leverage
- **Data Sources**: Multi-source (Yahoo, Binance, CoinGecko) with automatic failover
- **Risk Management**: Independent capital allocation, VaR limits, and circuit breakers
- **Notifications**: Real-time Discord/Telegram with heartbeat monitoring
- **Auto-Commit**: Enhanced system for artifacts and evaluation data

---

## 2) Quick start
- Install
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # set API keys, toggles
```
- Minimal run (no sends)
```
TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 \
python3 scripts/tracer_bullet_universe.py --config config/universe.yaml --top 10
```
- Evaluate (sample data included)
```
python3 scripts/run_eval_tests.py
python3 scripts/eval_ingest.py --input eval_data/resolved/sample.csv
python3 scripts/eval_runner.py
```
- Test enhanced ML features
```
python3 -c "
import sys
sys.path.insert(0, '.')
from scripts.hybrid_crypto_trader import _build_live_feature_vector
import pandas as pd
import numpy as np

# Test enhanced feature engineering
dates = pd.date_range('2023-01-01', periods=100, freq='15min')
test_df = pd.DataFrame({
    'open': np.random.uniform(50000, 60000, 100),
    'high': np.random.uniform(50000, 60000, 100),
    'low': np.random.uniform(50000, 60000, 100),
    'close': np.random.uniform(50000, 60000, 100),
    'volume': np.random.uniform(100, 1000, 100)
}, index=dates)

feature_names = [
    'ret_1', 'rsi', 'rsi_divergence', 'ema12', 'ema26', 'macd', 'macd_signal', 
    'macd_histogram', 'macd_momentum', 'sma20', 'std20', 'bb_upper', 'bb_lower', 
    'bb_position', 'bb_width', 'momentum_5', 'momentum_10', 'momentum_20', 
    'roc_5', 'roc_10', 'roc_20', 'atr', 'close_volatility', 'high_low_range', 
    'volume_sma', 'volume_ratio', 'volume_trend', 'price_acceleration', 
    'support_level', 'resistance_level', 'support_distance', 'resistance_distance', 
    'ema12_slope', 'ema26_slope', 'vol', 'vol_chg', 'cross_up'
]

result = _build_live_feature_vector(test_df, feature_names)
print(f'✅ Enhanced features working: {result.shape[1]} indicators computed')
"
```

---

## Important disclaimers

- This is a research/learning project. Not financial advice. Markets are volatile; use at your own risk.
- Experimental/testing phase: results may vary and won’t be perfectly consistent while guardrails are tuned.

---

## Tech stack used to build

- Perplexity API for LLM-based narrative synthesis (source‑tagged evidence lines)
- **NEW**: PyTorch with EnhancedMLP (batch normalization, dropout, early stopping) for advanced ML gating
- **NEW**: 37 technical indicators (RSI, MACD, Bollinger Bands, ATR, momentum, volume analysis, price patterns)
- Transformers + Torch for FinBERT and classifier/embedding primitives
- spaCy + sentence-transformers for relevance and semantic de‑duplication
- pandas + numpy for multi‑timescale features and divergence math
- Alpaca Trade API for market data and optional execution plumbing
- requests + httpx for robust HTTP integrations
- python‑dotenv + PyYAML + pydantic for clean config and schema validation
- python‑telegram‑bot (and Discord equivalent) for number‑free digests
- APScheduler for safe, staggered scheduling (opt‑in)
- matplotlib for quick visuals; pytest for sanity tests

Note: While examples focus on crypto (BTC/ETH), the framework generalizes to equities/ETFs/FX/rates.

---

## 3) Key features
- Multi‑asset universe scanning with crypto timeframes (1h/4h/1D/1W/1M).
- Evidence lines (concise, number‑free in chat), artifacts retain full metrics.
- Polymarket discovery via Perplexity Pro API; side‑by‑side internal vs market probability.
- Git auto-commit/push for artifacts (universe/evaluation), gated via env flags.
- Weekly evaluation wrapper and ingestion for resolved markets.

**Enhanced ML Trading System**:
- **37 Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, momentum, volume analysis, price acceleration, support/resistance levels
- **Advanced Neural Network**: EnhancedMLP with configurable layers (64-32-16), batch normalization, dropout (0.2), early stopping
- **ML Monitoring**: Comprehensive health tracking, drift detection, performance metrics, automatic retraining every 6 hours
- **Feature Synchronization**: Perfect parity between training and live trading feature pipelines
- **Risk Management**: ATR volatility filters, position sizing based on confidence, exploration parameters

Robust hybrid trader (opt-in gates):
- ML probability gate with PyTorch model (`eval_runs/ml/latest/model.pt`) and features parity.
- ATR volatility filter on 15m bars with min/max ATR% band.
- Higher-timeframe regime alignment via 1h EMA (configurable length).
- Optional heartbeat notifications every N runs.

---

## 4) Architecture map (files/directories)
- Core
  - `scripts/tracer_bullet_universe.py` — scan, enrich, digest
  - `config/universe.yaml` — asset universe
  - `universe_runs/` — JSON/CSV artifacts
- **Enhanced ML System**
  - `backtester/features.py` — 37 technical indicators with advanced feature engineering
  - `backtester/ml_baseline.py` — EnhancedMLP neural network with batch norm, dropout, early stopping
  - `backtester/ml_monitor.py` — comprehensive ML health monitoring and drift detection
  - `backtester/ml_gate.py` — ML probability gating for live trading
  - `scripts/ml_retrainer.py` — automatic model retraining every 6 hours
  - `eval_runs/ml/latest/` — latest trained model and feature metadata
- Digest delivery
  - `scripts/tg_sender.py`, `scripts/discord_sender.py` — safe split/chunk senders
  - `scripts/tg_digest_formatter.py`, `scripts/discord_formatter.py` — formatters
- Polymarket
  - `providers/polymarket_pplx.py` — Perplexity provider (key rotation, strict JSON parse)
  - `scripts/polymarket_bridge.py` — mapping, filters, internal probability calibration
- Evaluation (v3.4)
  - `scripts/eval_metrics.py` — Brier, log‑loss, calibration, cohorts
  - `scripts/eval_runner.py`, `scripts/eval_ingest.py`, `scripts/eval_weekly.py`
  - `eval_runs/` — per‑run outputs
- Ops
  - `autocommit.py` — stage/commit/push helper

---

## 5) Configuration (env flags)
- Universe git ops
  - `TB_UNIVERSE_GIT_AUTOCOMMIT=1`, `TB_UNIVERSE_GIT_PUSH=1`
- Evaluation git ops
  - `TB_EVAL_GIT_AUTOCOMMIT=1`, `TB_EVAL_GIT_PUSH=1`, `TB_EVAL_GIT_INCLUDE_DATA=1`
- Messaging safety
  - `TB_HUMAN_DIGEST`, `TB_NO_TELEGRAM`, `TB_NO_DISCORD`
- Polymarket
  - `TB_POLYMARKET_NUMBERS_IN_CHAT=0`, `TB_POLYMARKET_SHOW_EMPTY=1`, `TB_POLYMARKET_DEBUG=1`
  - Perplexity keys: `PPLX_API_KEY` or `PPLX_API_KEY_1..N` (model enforced to `sonar`)

**Enhanced ML Configuration**:
- `TB_USE_ML_GATE=1` — Enable ML probability gating
- `TB_ML_GATE_MODEL_PATH=eval_runs/ml/latest/model.pt` — Path to trained model
- `TB_ML_GATE_MIN_PROB=0.25` — Minimum probability threshold for trades
- `TB_ML_RETRAIN_EVERY_SEC=21600` — Retrain model every 6 hours
- `TB_ML_MONITOR_ENABLED=1` — Enable ML health monitoring
- `TB_ML_MODEL_TYPE=EnhancedMLP` — Neural network architecture
- `TB_ML_HIDDEN_DIMS=64,32,16` — Network layer dimensions
- `TB_ML_DROPOUT_RATE=0.2` — Dropout regularization
- `TB_ML_LEARNING_RATE=0.001` — Training learning rate

**Risk Management**:
- `TB_USE_ATR_FILTER=1` — Enable ATR volatility filtering
- `TB_ATR_MIN_PCT=0.002` — Minimum ATR percentage
- `TB_ATR_MAX_PCT=0.10` — Maximum ATR percentage
- `TB_TRADER_RISK_FRAC=0.000001` — Risk fraction per trade
- `TB_MAX_NOTIONAL_PER_TRADE=1000` — Maximum notional per trade

---

## 5.3) Enhanced ML Features (37 Technical Indicators)

The system now uses a comprehensive set of 37 technical indicators for superior ML predictions:

**Momentum & Trend Indicators**:
- RSI (Relative Strength Index) with divergence signals
- MACD (Moving Average Convergence Divergence) with momentum
- EMA slopes (12-period and 26-period)
- Multiple timeframe momentum (5, 10, 20 periods)
- Rate of Change (ROC) indicators

**Volatility & Range Indicators**:
- Bollinger Bands (position, width, upper/lower)
- ATR (Average True Range) proxy
- Close volatility (20-period rolling std)
- High-low range analysis

**Volume Indicators**:
- Volume SMA and ratio analysis
- Volume trend analysis
- Volume change signals

**Price Pattern Indicators**:
- Price acceleration (second derivative)
- Support/resistance level analysis
- Cross-up signals for EMA intersections

**ML Architecture**:
- EnhancedMLP neural network with configurable layers
- Batch normalization for training stability
- Dropout regularization (0.2 rate) to prevent overfitting
- Early stopping to avoid overtraining
- Automatic retraining every 6 hours with fresh data

All features are synchronized between training (`backtester/features.py`) and live trading (`scripts/hybrid_crypto_trader.py`) pipelines.

---

## 5.1) Payload schema v3.2 + Consistency gate

- Artifacts now include:
  - `evidence_line` (concise narrative for chat; numbers retained only in artifacts)
  - `thesis` snapshot with `action`, `risk_band`, `readiness`
  - Per‑TF `plan[tf]` with `entries`/`invalidation`/`targets`, plus `source` and `explain`
  - `timescale_scores.price_change_pct` (renamed from `price_move_pct`) and `alignment_flag`
  - Optional top‑level `polymarket[]`
- Deterministic consistency check (safe):
```
TB_DETERMINISTIC=1 TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 \
TB_UNIVERSE_GIT_AUTOCOMMIT=0 TB_UNIVERSE_GIT_PUSH=0 \
python3 scripts/consistency_check.py --config config/universe.yaml --top 10
```
Exit non‑zero on drift.

---

## 5.2) Nightly self‑checks & artifact auto‑commit

- Hit‑rate checks: compute 1h/4h/1D directional correctness by joining `universe_runs/*.json` with `bars/*.csv`.
- Tunables: `TB_HITRATE_SIDEWAYS_EPS` (sideways band), `TB_HITRATE_W_1H/_4H/_1D` (weighted vote), `TB_HITRATE_REG_THRESH` (regression warn).
- Nightly workflow runs in safe mode, appends `eval_runs/hit_rate_trend.csv`, compares vs previous, and auto‑commits non‑.py artifacts.
- Auto‑commit scope: stage all, then unstage `*.py` to ensure only JSON/CSV/MD/YAML land (e.g., `runs/*.json`, `universe_runs/metrics.csv`, `eval_runs/*`, `bars/*`).

Quick check (local):
```
python scripts/asset_hit_rate.py --runs_dir universe_runs --bars_dir bars --runs_map_dir runs \
  --debug --failures_csv eval_runs/hit_rate_failures.csv --markdown_out eval_runs/hit_rate_summary.md
```

---

## 6) Usage recipes
- Universe digest (no sends)
```
TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 \
python3 scripts/tracer_bullet_universe.py --config config/universe.yaml --top 10
```
- Polymarket discovery (debug)
```
TB_ENABLE_POLYMARKET=1 TB_POLYMARKET_DEBUG=1 \
python3 scripts/polymarket_bridge.py --max-items 4
```
- Weekly evaluation snapshot
```
python3 scripts/eval_weekly.py
```
- **NEW**: Test ML feature engineering
```
python3 -c "
from backtester.features import build_features
from backtester.core import DataLoader
import pandas as pd

# Load and test enhanced features
loader = DataLoader('bars/')
bars = loader.load()
X, y = build_features(bars)
print(f'✅ Features built: {X.shape[1]} indicators, {len(X)} samples')
print(f'Feature names: {list(X.columns)}')
"
```
- **NEW**: Check ML model health
```
python3 -c "
from backtester.ml_monitor import MLMonitor
monitor = MLMonitor()
health = monitor.get_model_health_score()
print(f'Current ML Health Score: {health:.3f}')
"

---

## 7) Non‑bias design (how we keep it honest)
- Objective tape vs narrative cross‑check
  - Tape: `alpaca.py`, `price.py`, `bars_stock.py`, `timescales.py`
  - Narrative: `perplexity_fetcher.py`, `pplx_fetcher.py`, `narrative_dev.py`
- Semantic relevance gating: `relevance.py`, `narrative_dev.py`
- Robust aggregation (MAD outlier drop, trimmed means): `sentiment_utils.py`
- Decay & timescale alignment: `narrative_dev.py`, `timescales.py`
- Confirmation checks (price vs narrative): `confirmation.py`, `timescales.py`
- Diversity & dedupe: `dedupe_utils.py`, `diversity.py`, `source_weights.py`, `debug_sources.py`
- Explainability: number‑free chat evidence; artifacts retain all metrics
- Continuous evaluation & calibration: `scripts/eval_metrics.py`, `scripts/eval_runner.py`, `scripts/eval_ingest.py`
- Ops guardrails: `autocommit.py`, `.env.example` (key rotation, push gating; retries/backoff WIP)

---

## 8) Initial data sources
- Market data (objective): Alpaca (`alpaca.py`, `price.py`, `bars_stock.py`)
- News synthesis (structured): Perplexity Pro API (`perplexity_fetcher.py`, `pplx_fetcher.py`)
- Optional mainstream feed: CoinDesk RSS (`coindesk_rss.py`)
- Prediction markets (reference): Polymarket via PPLX (`providers/polymarket_pplx.py`, `scripts/polymarket_bridge.py`)

---

## 9) Evaluation pipeline (v3.4)
- Inputs: `eval_data/resolved/*.csv` with `id,asset,title,closed_at,market_prob,internal_prob,outcome,cohort`
- Metrics: Brier, log‑loss, calibration bins (CSV), cohort win‑rates
- Outputs: `eval_runs/<timestamp>/metrics.json`, `calibration.csv`, `cohorts.csv`
- Git ops: `TB_EVAL_GIT_AUTOCOMMIT`, `TB_EVAL_GIT_PUSH`, `TB_EVAL_GIT_INCLUDE_DATA`

---

## 10) Digest delivery (Telegram/Discord)
- Telegram: safe splitter, crypto‑only mode (`TB_DIGEST_TELEGRAM_CRYPTO_ONLY=1`), respects `TB_HUMAN_DIGEST`/`TB_NO_TELEGRAM`
- Discord: chunked embeds, respects `TB_NO_DISCORD`
- Number gating for chat: `TB_POLYMARKET_NUMBERS_IN_CHAT=0`

Parity highlights (v3.1.16):
- A+ setups appear as `[A+ Setup]` in headers and `(A+)` in Quick Summary coins on both Telegram and Discord.
- Kid‑friendly Quick Summary at the end of digests.
- Plain‑English phrasing applied to Executive Take, Weekly Plan, and Engine Thesis.

---

## 11) Troubleshooting
- Empty Polymarket section: check PPLX keys and `TB_POLYMARKET_DEBUG=1`
- No sends: ensure `TB_NO_TELEGRAM=0`/`TB_NO_DISCORD=0`, valid chat tokens
- Git push blocked: unset `*_PUSH` or fix remote; commits still land locally when `*_AUTOCOMMIT=1`
- **NEW**: ML model issues:
  - Model not loading: check `TB_ML_GATE_MODEL_PATH` points to valid `.pt` file
  - Feature mismatch: ensure training and live features are synchronized (37 indicators)
  - Poor predictions: check `ml_monitor.log` for health score and retrain if needed
  - Memory issues: reduce `TB_ML_HIDDEN_DIMS` or increase system memory
- **NEW**: Feature engineering errors:
  - NaN values: check input data quality and ensure sufficient historical bars
  - Shape mismatches: verify feature_names list matches between training/live pipelines

---

## 12) License
See `LICENSE`.

---

## 13) Live Hybrid Trading Agent — Status and Ops (2025-09-01)

This repository includes a hybrid trading agent that is currently live, autonomous, and self-learning with guardrails.

### Current status
- **Enhanced ML System**: Now running with 37 technical indicators, EnhancedMLP neural network, and comprehensive monitoring
- Live processes: `scripts/start_hybrid_loop.sh` runs a wrapper that manages a periodic ML retrainer and a resilient trader loop (`python3 scripts/hybrid_crypto_trader.py`).
- Guardrails:
  - Watchdog cron restarts the loop if it dies.
  - Daily health check verifies logs freshness, recent runs, and promoted params; includes a self-heal path.
  - Weekly propose+canary refresh evaluates new parameter proposals and only promotes upon canary pass.
  - **NEW**: ML health monitoring with drift detection and automatic model retraining every 6 hours.
- Artifacts: Non-code artifacts (JSON/CSV/MD/YAML, images) are auto-committed and pushed. `.py` files are never auto-committed.

### Crons installed (user crontab)
```
*/2 * * * * /bin/bash -lc "/Users/mouryadamarasing/Documents/Project-Tracer-Bullet/scripts/watchdog_hybrid.sh" # com.tracer.watchdog-hybrid
0 9 * * * /bin/bash -lc "/Users/mouryadamarasing/Documents/Project-Tracer-Bullet/scripts/health_check.sh" # com.tracer.health-check
0 3 * * 0 /bin/bash -lc "/Users/mouryadamarasing/Documents/Project-Tracer-Bullet/scripts/weekly_propose_canary.sh" # com.tracer.weekly-propose-canary
0 3 * * 3 /bin/bash -lc "/Users/mouryadamarasing/Documents/Project-Tracer-Bullet/scripts/weekly_propose_canary.sh" # com.tracer.weekly-propose-canary-backup
```

### Health/self-heal behavior
- `scripts/health_check.sh` flags when `config/promoted_params.json` is missing/stale and will attempt a single self-heal by running `scripts/weekly_propose_canary.sh` (lock-protected to avoid spam), then re-checks freshness before alerting.
- Alerts are gated by environment:
  - Discord only if `TB_ENABLE_DISCORD=1` and `DISCORD_WEBHOOK_URL` present.
  - Telegram is disabled when `TB_NO_TELEGRAM=1`.

### Artifact policy (permanent directive)
- Auto-commit/push all non-code artifacts after runs to keep the working tree clean and maintain traceability.
- Never auto-commit `.py` files. Documentation updates are committed automatically.

### Operational commands (manual)
- Verify crons: `crontab -l`
- Manual watchdog trigger: `bash scripts/watchdog_hybrid.sh`
- Manual health check (no sends): `TB_ENABLE_DISCORD=0 TB_NO_TELEGRAM=1 bash scripts/health_check.sh`
- Refresh backtest rollups: `python3 scripts/backtest_aggregate.py --out_root eval_runs/backtests`

### Validation tests performed (2025-09-01)
- **Enhanced ML Features**: Successfully validated 37 technical indicators computation with synchronized training/live pipelines
- **Feature Vector Test**: Confirmed torch.Size([1, 37]) output from `_build_live_feature_vector()`
- Watchdog test: killed the trader once; watchdog/wrapper ensured the loop remained healthy and trader kept running.
- Health self-heal dry-run: moved `config/promoted_params.json` aside, ran `scripts/health_check.sh` with sends/commits disabled; it executed `scripts/weekly_propose_canary.sh` (auto-tuner + canary). No promotion occurred; original params restored. Artifacts were written under `eval_runs/auto_tuner/<ts>/` and `eval_runs/canary/<ts>/notify.txt`.
- Backtest rollups: refreshed `eval_runs/backtests/aggregate.md` and `aggregate.csv` after the batch. These will be kept current post-batches.

### ML Health Monitoring System
- **Performance Tracking**: Monitors accuracy, precision, recall, F1-score, and AUC metrics
- **Drift Detection**: Identifies when model performance degrades over time
- **Health Scoring**: Composite health metric combining multiple performance indicators
- **Automatic Retraining**: Triggers model retraining when health score falls below threshold
- **Logging**: Comprehensive logging to `ml_monitor.log` with structured metrics
- **Alerting**: Discord/Telegram notifications for critical ML health issues

### Quick references
- Logs: `trader_loop.log`, `trader_loop.err`, `ml_monitor.log`
- Artifacts: `runs/…`, `eval_runs/…` (auto-committed, non-code only)
- Parameters: `config/promoted_params.json`
- Latest model link: `eval_runs/ml/latest/`
- ML Health Dashboard: Check `backtester/ml_monitor.py` for current health status

