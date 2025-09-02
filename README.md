Project Confidence Engine
Here is a draft for the README introduction that puts the Tracer Bullet philosophy and mission front and center, clearly distinguishing it from generic sentiment bots. This will ensure anyone reading knows the core vision and agile process powering your divergence engine:

***

# Confidence Engine — Institutional-Grade Multi-Asset Divergence Engine
> Looking for the concise version? See [README_CLEAN.md](README_CLEAN.md) | [Enhanced Features Summary](ENHANCED_HYBRID_SUMMARY.md) | [Dual Agent System](DUAL_AGENT_README.md)

## 🚀 Latest: Dual-Agent System with Futures & Perpetuals Trading

**NEW**: Complete dual-agent architecture with independent risk management:

- **🤖 Main Agent (Low-Risk)**: Enhanced hybrid agent with 20+ crypto assets, ML gates, and adaptive strategies
- **⚡ Futures Agent (High-Risk)**: New leveraged futures/perpetuals agent with 5x-25x leverage and momentum trading
- **📊 Unified Monitoring**: Real-time dashboard tracking both agents
- **🛡️ Independent Operation**: Each agent manages its own capital and risk
- **📈 Free Paper Trading**: Test futures strategies without cost

**Status**: Both agents operational | **Main Agent**: Live trading | **Futures Agent**: Live trading | **Monitoring**: Real-time dashboard active

### Current Live Configuration (September 2, 2025)
- **Main Agent**: ✅ Live trading with 15 crypto assets, ML gates, ATR filters, and adaptive thresholds
- **Futures Agent**: ✅ Live trading with 20 blue chip futures, 25x leverage cap, $100 margin cap
- **Platforms**: Binance Futures (testnet) with real API integration and proper precision handling
- **Risk Management**: Dynamic leverage based on risk-reward ratios, volatility adjustments, hard caps
- **Database**: SQLite auto-commit enabled, enhanced_trading.db tracked in GitHub
- **Notifications**: Discord and Telegram with trade confirmations and heartbeat monitoring
- **Status**: Both loops operational, 9+ trades executed, all systems stable

## Our Mission: Exploit Emotional-Math Gaps with Perfect Discipline

Financial markets are driven by human emotion as much as by fundamentals. Crowd fear, greed, and narrative bias often cause prices to diverge significantly from their true, statistical values — creating opportunities for disciplined, rational traders.

**Confidence Engine is not just another sentiment bot. It is a purpose-built divergence engine, designed to detect the market's "lie" between story and price, and deliver actionable alpha signals in real time.**

We ask the key question nonstop:

> *"The news narrative is strongly positive (or negative), but why hasn't the price moved yet? Is the market underreacting or overreacting?"*

By measuring and quantifying these divergences rigorously, filtering for relevance and confidence, and adapting thresholds dynamically, Confidence Engine provides **real-time, explainable**, and **action-ready** trading signals — all delivered with evidence and playbook guidance directly to your Telegram DM.

## Our Mission: Exploit Emotional-Math Gaps with Perfect Discipline

Financial markets are driven by human emotion as much as by fundamentals. Crowd fear, greed, and narrative bias often cause prices to diverge significantly from their true, statistical values — creating opportunities for disciplined, rational traders.

**Confidence Engine is not just another sentiment bot. It is a purpose-built divergence engine, designed to detect the market’s “lie” between story and price, and deliver actionable alpha signals in real time.**

We ask the key question nonstop:

> *"The news narrative is strongly positive (or negative), but why hasn’t the price moved yet? Is the market underreacting or overreacting?"*

By measuring and quantifying these divergences rigorously, filtering for relevance and confidence, and adapting thresholds dynamically, Confidence Engine provides **real-time, explainable**, and **action-ready** trading signals — all delivered with evidence and playbook guidance directly to your Telegram DM.

## Why Confidence Engine Is Now Institutional-Grade

- **Dual-Agent Architecture:** Independent main agent (low-risk spot trading across 20+ assets) and futures agent (high-risk leveraged trading with 5x-25x leverage) for diversified risk management
- **Multi-Asset Portfolio:** Advanced portfolio management across BTC/USD, ETH/USD, SOL/USD, LINK/USD, and futures markets with correlation controls  
- **Futures Trading Platforms:** Multiple free platforms including Binance Futures Testnet, Bybit Futures Demo, BitMEX Futures Testnet, Deribit Futures Test for comprehensive market coverage
- **Advanced Risk Management:** Portfolio VaR limits, individual position limits, regime-based risk adjustments, and independent capital allocation ($10k for futures agent)
- **Adaptive Learning:** Real-time parameter optimization with exploration windows and performance tracking for both agents
- **Philosophy-first:** We shipped a fast, end‑to‑end prototype (V1) to validate our core edge early, avoiding premature complexity.
- **Elegant simplicity:** We leverage two clean, high-quality signal pillars — Alpaca price data and Perplexity synthesized narrative — before adding noisy social feeds.
- **Hybrid intelligence:** Combining quantitative “oracle” data and qualitative “psychologist” narrative to find true market inefficiencies.
- **Divergence over sentiment:** Unlike traditional news sentiment bots, we trade the **gap** between narrative and price, where predictable emotional mistakes happen.
- **Adaptive and auditable:** Confidence shaping, source weighting, catalyst tagging, and full payload provenance ensure discipline and continuous improvement.
- **Generalizes beyond crypto:** While BTC/ETH are emphasized today, the approach extends to equities, ETFs, FX, rates, and other markets; Polymarket is optional, not required.

## How We Work

- Ingest multi-source crypto news with semantic filtering for asset relevance.
- Score narrative sentiment versus price/volume context.
- Compute divergence signal with adaptive thresholds.
- Generate concise alpha-first summaries and next steps.
- Deliver signals and evidence automatically via Telegram DMs.
- Persist rich data for audit, backtesting, and iterative tuning.

## Enhanced Feature Engineering: 37 Technical Indicators

The Confidence Engine now includes a comprehensive set of 37 technical indicators for robust ML model inputs:

### Momentum Indicators (13)
- RSI (Relative Strength Index) - 14, 21, 28 periods
- MACD (Moving Average Convergence Divergence) - momentum, signal, histogram
- Stochastic Oscillator - %K, %D
- Williams %R
- Momentum (10, 20 periods)
- ROC (Rate of Change) - 5, 10, 20 periods

### Trend Indicators (8)
- EMA (Exponential Moving Averages) - 12, 26, 50, 200 periods
- SMA (Simple Moving Averages) - 20, 50 periods
- Trend Strength - slope calculations
- ADX (Average Directional Index)

### Volatility Indicators (6)
- ATR (Average True Range) - 14, 21 periods
- Bollinger Bands - width, %B
- Standard Deviation - 20 period
- Volatility Ratio

### Volume Indicators (5)
- Volume Z-Score
- Volume Ratio
- OBV (On Balance Volume)
- Volume Momentum
- VWAP (Volume Weighted Average Price)

### Support/Resistance & Price Action (5)
- Pivot Points
- Fibonacci Retracement levels
- Support/Resistance levels
- Price Acceleration
- Gap Analysis

### Live Trading Synchronization
- **Feature Vector Building**: Updated `scripts/hybrid_crypto_trader.py` `_build_live_feature_vector()` to compute all 37 indicators
- **Validation**: Successfully tested with torch.Size([1, 37]) and all features computed correctly
- **Consistency**: Ensures training and inference use identical feature engineering

## ML Architecture & Monitoring

### Enhanced Model Architecture
- **Layers**: 64-32-16 configuration with batch normalization and dropout
- **Training**: Early stopping, comprehensive metrics tracking, improved convergence
- **PyTorch Implementation**: Robust neural network for probability gating

### ML Gate Integration
- **Probability Thresholds**: Configurable BUY gating based on model confidence
- **Model Loading**: Automatic loading from `eval_runs/ml/latest/model.pt`
- **Health Monitoring**: Performance tracking, drift detection, model degradation alerts
- **Safety Gates**: Soft gate mode for inference failures, minimum probability floors

### Comprehensive Monitoring System
- **Performance Tracking**: Prediction logging, model health scoring
- **Drift Detection**: Automatic detection of model performance degradation
- **Health Metrics**: Calibration analysis, prediction confidence monitoring

## Risk Management Enhancements

### Position Sizing
- **Confidence-Based Sizing**: Dynamic position sizing based on signal confidence
- **Risk Controls**: Per-trade notional caps, volatility-adjusted sizing
- **ATR-Based Stops**: Dynamic stop-loss sizing with configurable multipliers

### Safety Gates
- **Notional Caps**: Hard per-trade USD limits (`TB_MAX_NOTIONAL_PER_TRADE`)
- **Volatility Filters**: ATR-based market condition filtering
- **HTF Regime Checks**: Higher-timeframe trend alignment requirements

## Autonomous Operations Framework

### Loop Management
- **Start Script**: `scripts/start_hybrid_loop.sh` with auto-apply promoted parameters
- **ML Retraining**: Periodic model retraining with artifact management
- **Artifact Auto-Commit**: Automatic commit of artifacts with git integration

### Health Monitoring
- **Watchdog System**: Cron-based process monitoring with automatic restart
- **Health Checks**: Comprehensive system status verification
- **Parameter Tuning**: Weekly canary runs with backtest validation

### Notification System
- **Multi-Channel**: Telegram and Discord notifications
- **Heartbeat Monitoring**: Optional liveness notifications
- **Event-Driven Alerts**: Critical system event notifications

***

## Important disclaimers

- Not financial advice. Research/learning tool; use at your own risk.
- Experimental/testing phase: outputs can vary and may not be perfectly consistent while guardrails are tuned.

***

Summary
An always-on research and trading agent that spots short-term mispricings by comparing what the story says (news-driven narrative momentum) to what the tape shows (price momentum). When those disagree in a meaningful, well-defined way, the agent takes a paper trade with full explainability, logs everything to a local database, and sends a Telegram “style analysis” message you can read in one glance.

Philosophy (my version)
- Mastery through evidence: Every idea is a hypothesis with a falsifiable test. We promote or kill based on data.
- Thin thread first: Always get an end-to-end path running before adding features. Complexity earns its way in.
- Explainability as a feature: If we can’t justify a signal in one paragraph, we don’t trade it.
- Two independent eyes: LLM narrative and FinBERT sentiment cross-check each other; TA/volume validates the tape side.
- Time is a feature: Narrative impact decays; events matter; regimes change. We model decay and use event triggers.
- Reliability beats cleverness: Strict schemas, retries, key rotation, circuit breakers, and idempotent orders protect edge.
- Anti-overfit protocols: Small, interpretable feature set; walk-forward validation; cohort analytics; no indicator soup.
- Incremental risk: Paper → tiny live → scale where evidence persists.
- Human-in-the-loop: Telegram “style analysis” for every signal; review on cadence, not mid-drawdown emotion.
- Build your own dataset: The real moat is the corpus of divergence signals with reasons and outcomes.

Core Idea
- Narrative Momentum: A score from −1 to +1 derived from LLM narrative polarity/confidence on recent headlines, blended with FinBERT sentiment, discounted by time decay. New, credible stories matter more; old ones fade.
- Price Momentum: A score from −1 to +1 from simple, interpretable indicators: RSI, short-vs-medium MA slope, MACD histogram (acceleration), volume z-score (participation), with optional ATR context.
- Divergence: narrative_z − price_z. Large positive: price may be lagging good story. Large negative: price may be lagging bad story. Only act when confidence and risk checks pass.

Tech Stack (and why)
- Language/IDE: Python 3.10+ in Cursor AI (GPT-5) for rapid scaffolding, refactors, and TDD support.
- Market Data & Trading: Alpaca (paper) for prices, headlines, and execution. Free, stable, and live-parity API.
- LLM/Narrative: Perplexity Pro API for production narrative synthesis (you already have keys). In dev, a local adapter to iterate prompts without burning credits.
- NLP: FinBERT (ProsusAI) via Hugging Face for financial sentiment on headlines; spaCy for keywords/entities.
- Analytics: pandas/numpy for indicators and normalization; APScheduler for periodic/event-driven runs.
- Storage/Logging: SQLite for durable local logs of prompts, signals, trades, and raw payloads.
- Notifications: python-telegram-bot for real-time signal messages with “style analysis.”
- Testing/Quality: pytest, coverage, black/isort/ruff, pre-commit, python-dotenv for configuration.

Why this stack: It’s zero-cost for V1, reliable, widely used, and easy to harden. Clean upgrade paths exist for streaming, dashboards, and deployment.

Architecture
- Data
  - alpaca_fetcher: bars and headlines from Alpaca; optional yfinance for backtests.
- Narrative Adapters
  - narrative_local: development adapter (Cursor) for prompt/schema iteration.
  - narrative_perplexity: production adapter with key rotation, retries, timeouts, strict JSON validation, and repair-then-reject.
- NLP
  - sentiment_finbert: batch headline sentiment → continuous score in [−1, +1].
  - keyword_extractor: entities/keywords for explainability and novelty.
- Analysis
  - price_momentum: RSI, MA slope (e.g., 10 vs 50), MACD histogram, volume z-score, optional ATR context → scaled to [−1, +1].
  - narrative_momentum: LLM polarity + confidence blended with FinBERT; exponential time decay with configurable half-life.
  - divergence: z-score normalization and divergence calculation; thresholds and reason codes.
- Execution & Risk
  - trade_executor: Alpaca paper orders with unique client order IDs for idempotency; market-first with optional limits later.
  - risk: per-symbol cooldown, max positions, min volume credibility, timed exits.
- Orchestration
  - agent: APScheduler job to run scans; optional event triggers on headline deltas; graceful error handling and degradation paths.
- Persistence & Notifs
  - db_manager: SQLite tables for prompts, signals, trades; raw JSON blobs preserved.
  - telegram_notifier: structured “style analysis” message for each signal; optional heartbeat.

Data Contracts
Narrative JSON (must validate before use)
- narrative_summary: string
- narrative_momentum_score: float in [-1, 1]
- confidence: float in 
- salient_entities: list of strings
- anchor_quotes: list of short strings
- metadata (optional): { source_sample: list[str], model_name: str, created_at: iso8601 }

Parser policy: validate against a typed schema; attempt one repair for minor JSON issues; if still invalid, skip trading and log the sample.

Decision Blueprint
- Inputs:
  - Narrative Score = weighted blend of LLM polarity and FinBERT sentiment, decayed over time.
  - Price Score = composite from normalized RSI, MA slope, MACD histogram; down-weight if volume z-score is weak.
- Divergence:
  - divergence = z(narrative) − z(price).
- Trigger (initial defaults, to be tuned):
  - abs(divergence) > 1.0,
  - narrative confidence > 0.6,
  - volume credibility floor (e.g., volume z > −0.5),
  - cooldown satisfied.
- V1: long-only. Shorts considered later after backtesting and guardrails.
- Explainability: store the summary, key quotes, indicators, scores, thresholds, and reason_code for each signal.

Reliability
- API failure handling: retries with backoff, Perplexity key rotation, timeouts, and circuit breakers to pause narrative calls if failure rate spikes.
- Degradation paths:
  - Narrative down → TA-only mode or skip entries.
  - Market data down → pause trading; keep logging narratives if available.
- Orders: unique client order IDs avoid duplicate fills on retry.
- Alerts: Telegram on critical errors and circuit-breaker events.

Backtesting Integrity
- Event-ordered replay: process bars and headlines strictly by timestamp.
- Point-in-time features only: lag everything properly; no future info in current decisions.
- Validation: in-sample tuning vs. out-of-sample testing; walk-forward to mirror deployment; robustness checks to small parameter changes.
- Cohorts: measure by asset, time-of-day, volatility regime, and event type.

Roadmap
V1 — Tracer Bullet (single-asset E2E)
- One symbol (recommend BTCUSD for 24/7 flow).
- Fetch last 60m bars and latest headlines.
- Dev narrative adapter returns strict JSON; schema-validated and logged.
- Compute RSI, MA slope, MACD histogram, volume z-score → Price Score.
- Blend narrative + FinBERT with decay → Narrative Score.
- Compute divergence; print decision preview; then paper trade under thresholds.
- Telegram “style analysis” message for each signal.
Acceptance: stable end-to-end run that can place a paper trade and log full context.

V2 — Better Signals
- Event-driven scans on headline deltas (detect novelty/changes).
- Narrative decay active; basic novelty weighting to avoid echo chasing.
- Multi-symbol universe (10–20 news-sensitive names).
- Enhanced explainability in Telegram and DB.
Acceptance: fewer, higher-quality signals; readable evidence per trade.

V3 — Backtesting and Governance
- Event-driven backtester with leak-free design.
- Walk-forward parameter tuning; monthly governance cadence.
- Cohort metrics by regime and event type.
Acceptance: documented thresholds from out-of-sample results and reproducible runs.

V4 — Execution Quality
- Microstructure-aware tweaks: limit orders when spreads widen; volatility-aware position sizing; stricter cooldowns.
Acceptance: reduced slippage and improved realized PnL on paper.

V5 — Data Breadth and Explainability+
- Optional crowd sentiment as secondary attention proxy (filtered).
- Source credibility weighting learned over time.
- Per-signal “case file” artifact.
Acceptance: improved precision and faster postmortems.

V6 — Small Live Capital and Safety
- Small live deployment with strict caps and daily loss limits.
- Heartbeats, daily summaries, anomaly alerts; version tagging on all signals.
Acceptance: stable live operation with rollback path.

Setup
1) Prereqs
- Python 3.10+
- Alpaca paper trading account (API key/secret)
- Perplexity Pro API keys (comma-separated list for rotation)
- Telegram bot token and chat_id

2) Environment
- Clone repo, create a venv, install requirements.
- Copy .env.example → .env; fill ALPACA, PERPLEXITY, TELEGRAM values.

3) First Run (preview mode)
- Run tracer_bullet.py for BTCUSD. It fetches prices/headlines, produces narrative JSON via dev adapter, computes TA and divergence, and prints a decision preview. No orders yet.

4) Paper Trading
- Flip the execution flag to enable Alpaca paper orders once previews look sane.
- Watch SQLite and Telegram messages for “style analysis” evidence.

Repository Structure
- agent.py (scheduler + orchestration)
- tracer_bullet.py (single-asset E2E tracer)
- src/
  - adapters/ narrative_base.py, narrative_local.py, narrative_perplexity.py
  - data/ alpaca_fetcher.py, yfinance_fetcher.py
  - nlp/ sentiment_finbert.py, keyword_extractor.py
  - analysis/ price_momentum.py, narrative_momentum.py, divergence.py
  - execution/ trade_executor.py, risk.py
  - db/ db_manager.py, schema.sql
  - utils/ logging_setup.py, retry.py, time_utils.py
- tests/
  - test_parsers.py, test_price_momentum.py, test_divergence.py, test_db.py
- docs/
  - roadmap.md, knowledge_wiki.md, dev_log.md

Schemas (SQLite)
- signals(id, ts, symbol, narrative_score, price_score, divergence, confidence, action, reason_code, json_blob, version_tag)
- trades(id, signal_id, ts, side, qty, fill_price, status, pnl_15m, pnl_60m)
- prompts(id, ts, symbol, prompt_hash, model_name, tokens_in, tokens_out, latency_ms)

Security and Safety Notes
- Never trade on invalid JSON parses; repair then reject.
- Respect rate limits; rotate Perplexity keys; exponential backoff.
- Circuit-breaker halts on repeated failures; send Telegram alert.
- Parameter changes only on a scheduled cadence after review.

Getting Help
- If something fails, check logs/ and the SQLite tables for the latest signal with reason_code.
- For Telegram issues, verify bot token and chat_id, and ensure your bot has started a conversation with you.
- For Alpaca issues, test account endpoints (account, clock) before trading.

License
 MIT for flexibility.
----------------------------
Update Tracer Bullet V1

Filename: README_Agent_V1.md
```markdown
# Tracer Bullet V1 – README (Consolidated)

This document consolidates architecture, configuration, runbook, file summaries, dev log, roadmap, changelog, and a suggested commit message for the Agent V1 release.

---

## 1) Overview

Agent V1 ingests multi-source crypto headlines (Perplexity Pro API with key rotation, CoinDesk RSS, Alpaca), deduplicates, semantically filters for BTC relevance, computes robust FinBERT sentiment on relevant-only headlines, builds a narrative from accepted headlines with confidence heuristics, applies decay, scores price/volume, computes divergence, and makes a guarded decision with an adaptive trigger. It exports JSON, bars CSV, and accepted headlines TXT, then auto-commits artifacts.

Key features:
- Multi-source ingest with Perplexity key rotation
- Robust sentiment: MAD outlier removal + 10% trimmed mean
- Narrative built from accepted headlines; conservative confidence fallback
- Adaptive divergence trigger (volume-aware)
- Provenance tagging for accepted headlines
- Exports: JSON payload, bars CSV, accepted headlines TXT
- Debug visibility: accepted (source, score) + relevance top-5

---

## 2) Architecture

- Ingest
  - Alpaca: latest_headlines(symbol, limit)
  - Perplexity Pro API: sonar-pro chat-completions; multiple API keys rotation
  - CoinDesk RSS (optional)
  - Deduplication across sources (preserve original text)
- Relevance gating
  - Semantic similarity against enriched BTC topic (RELEVANCE_THRESHOLD ~0.40–0.45)
  - Keyword fallback if 0 accepted
- Sentiment + Narrative
  - FinBERT on relevant-only headlines
  - Robust aggregation: MAD-based outlier drop → 10% trimmed mean
  - Narrative built from accepted-only; confidence: 0.55 (1), 0.65 (2+)
  - Composite: blend narrative + FinBERT; decay with staleness (half-life)
- Price/Volume + Divergence
  - Price score from bars; volume Z
  - Divergence = decayed narrative − price score
  - Adaptive trigger based on volume participation
- Decision
  - BUY/HOLD only (guardrails: confidence cutoff, divergence trigger, volume floor)
- Exports + Dev tooling
  - JSON payload, bars CSV, accepted TXT
  - Auto-commit to repo
  - Debug scripts for env/keys and source sanity

---

## 3) File-by-file summaries

- tracer_bullet.py
  - End-to-end orchestration: ingest (Alpaca + Perplexity rotation + CoinDesk), dedupe, relevance gating, robust sentiment, narrative build/decay, price/volume, divergence, adaptive decision, exports (JSON/CSV/TXT), auto-commit, console preview.
  - Includes provenance tagging and debug prints (accepted sources/scores, relevance top-5).

- finbert.py
  - FinBERT scoring utilities.
  - sentiment_robust: per-headline pos-neg in [-1,1], MAD outlier drop, 10% trimmed mean; returns (aggregate, kept_scores, dropped_outliers).

- sentiment_utils.py
  - trimmed_mean(values, trim), mad(a), drop_outliers(values, z_thresh) via robust MAD z-scores.

- narrative_dev.py
  - filter_relevant(headlines, threshold): semantic BTC relevance
  - make_from_headlines(accepted_only): narrative summary from accepted; confidence heuristic; neutral baseline score (composite blending used downstream).

- pplx_fetcher.py
  - Perplexity chat-completions (sonar-pro) client with multi-key rotation; returns (titles, items, err) with strict JSON parsing.

- coindesk_rss.py
  - RSS fetcher with retry/backoff; returns title list.

- dedupe_utils.py
  - Normalization + dedupe; preserves original capitalization.

- provenance.py
  - Tag accepted headlines with source: perplexity | alpaca | coindesk | unknown.

- narrative_analysis_extras.py
  - adaptive_trigger(base_trigger, volume_z): volume-aware divergence trigger; clamps to [0.6, 1.5].

- export.py
  - export_run_json(run_id, payload): runs/<id>.json
  - save_bars_csv(run_id, bars): bars/<id>.csv
  - save_accepted_txt(run_id, accepted_with_src): runs/<id>_accepted.txt

- config.py
  - .env settings + parsing. Supports:
    - PPLX_API_KEY_1..N (preferred)
    - PPLX_API_KEYS (comma-separated)
    - PPLX_API_KEY (single)
  - USE_COINDESK, PPLX_ENABLED, thresholds, lookbacks.

- debug_sources.py
  - Dev: print counts/samples from Alpaca, Perplexity (rotation), CoinDesk.

- test_pplx_auth.py
  - Dev: HTTP 200/401 check per Perplexity key.

- inspect_env_pplx.py
  - Dev: raw env inspection for PPLX_API_KEYS to detect hidden characters.

---

## 4) Configuration (.env)

Perplexity keys (preferred numbered variables):
```
PPLX_ENABLED=true
PPLX_API_KEY_1=pk_live_keyA
PPLX_API_KEY_2=pk_live_keyB
PPLX_API_KEY_3=pk_live_keyC
PPLX_API_KEY_4=pk_live_keyD
PPLX_HOURS=24
```

## Project map (what lives where)

- **Core engine**
  - `tracer_bullet.py` — end-to-end scan and messaging (earlier versions)
  - `scripts/tracer_bullet_universe.py` — multi-asset scan, digest build, TG/Discord send
  - `config/universe.yaml` — symbols and settings for scans
  - `universe_runs/` — enriched universe artifacts (JSON, CSV)

- **Digest delivery**
  - `scripts/tg_digest_formatter.py`, `scripts/tg_weekly_engine.py` — digest assembly (crypto TFs, weekly/engine minute)
  - `scripts/tg_sender.py` — Telegram safe splitter + sender (respects `TB_HUMAN_DIGEST`/`TB_NO_TELEGRAM`)
  - `scripts/discord_formatter.py`, `scripts/discord_sender.py` — Discord embeds with chunking (respects `TB_NO_DISCORD`)

- **Polymarket (BTC/ETH/SOL/XRP)**
  - `providers/polymarket_pplx.py` — Perplexity Pro API provider (key rotation, strict JSON parse)
  - `scripts/polymarket_bridge.py` — mapping, filtering, internal probability calibration

- **Evaluation (v3.4)**
  - `scripts/eval_metrics.py` — Brier, log‑loss, calibration, cohorts
  - `scripts/eval_runner.py` — compute metrics from `eval_data/resolved/*.csv` → `eval_runs/<ts>/`
  - `scripts/eval_ingest.py` — append resolved rows into monthly CSVs (`eval_data/resolved/YYYYMM.csv`)
  - `scripts/eval_weekly.py` — weekly wrapper to run evaluator
  - `eval_runs/` — evaluation outputs per run (metrics.json, calibration.csv, cohorts.csv)

- **Ops & reliability (v4.3)**
  - `autocommit.py` — stage/commit/push helper used by universe + evaluation
  - Flags across senders/providers for retries/backoff and degraded‑run behavior

***

## Evaluation pipeline (v3.4)

### Inputs
- Source CSVs at `eval_data/resolved/*.csv` (often monthly via `eval_ingest.py`)
- Required columns: `id,asset,title,closed_at,market_prob,internal_prob,outcome,cohort`

### Metrics computed
- Brier score, log‑loss
- Calibration bins (to CSV), with optional plot (can be added)
- Cohort win‑rates (by asset or provided cohort)

### Usage
- Run unit tests:
```
python3 scripts/run_eval_tests.py
```
- Ingest resolved rows (dedup, grouped by closed_at month):
```
python3 scripts/eval_ingest.py --input path/to/resolved_batch.csv
```
- Run evaluator (writes `eval_runs/<ts>/`):
```
python3 scripts/eval_runner.py
```
- Weekly wrapper (same as runner, convenient entrypoint):
```
python3 scripts/eval_weekly.py
```

### Auto‑commit/push of eval artifacts
- Controlled by env flags (see below). When enabled, `eval_runner.py` and `eval_ingest.py` will stage/commit and optionally push outputs:
  - `TB_EVAL_GIT_AUTOCOMMIT=1`
  - `TB_EVAL_GIT_PUSH=1`
  - `TB_EVAL_GIT_INCLUDE_DATA=1` to include `eval_data/resolved/` in the same commit

***

## Auto‑commit & Git ops (universe + evaluation)

- Universe artifacts (JSON/CSV) — set in `.env`:
  - `TB_UNIVERSE_GIT_AUTOCOMMIT=1`
  - `TB_UNIVERSE_GIT_PUSH=1`
  - Artifacts: `universe_runs/…`, plus optional mirrors to `runs/`

- Evaluation artifacts (see above) — set in `.env`:
  - `TB_EVAL_GIT_AUTOCOMMIT=1`
  - `TB_EVAL_GIT_PUSH=1`
  - `TB_EVAL_GIT_INCLUDE_DATA=1` (optional)

Commit messages are prefixed (e.g., `eval: metrics <ts>`), and pushes are gated by the `*_PUSH` flags.

***

## Hit‑rate self‑checks & nightly automation

- Asset hit‑rate checks compute 1h/4h/1D directional correctness by joining `universe_runs/*.json` with `bars/*.csv`.
- Tunables:
  - `TB_HITRATE_SIDEWAYS_EPS` (default 0.1): sideways band in percent.
  - `TB_HITRATE_W_1H`, `TB_HITRATE_W_4H`, `TB_HITRATE_W_1D` (default 1.0): horizon weights for weighted vote.
  - `TB_HITRATE_REG_THRESH` (default 0.05): nightly regression threshold.
- Nightly workflow `.github/workflows/safe_qa_nightly.yml` runs in safe mode, appends trend to `eval_runs/hit_rate_trend.csv`, logs regression compare, and commits non‑.py artifacts.
- Auto‑commit scope in nightly: stages all changes and then unstages `*.py`, ensuring artifacts like `runs/*.json`, `universe_runs/metrics.csv`, `eval_runs/*`, and `bars/*` are pushed.

Quick CLI:
```
python scripts/asset_hit_rate.py --runs_dir universe_runs --bars_dir bars --runs_map_dir runs --debug \
  --failures_csv eval_runs/hit_rate_failures.csv --markdown_out eval_runs/hit_rate_summary.md
```

***

## Payload schema v3.2 (artifact highlights)

- Evidence line: per‑asset `evidence_line` persisted for explainability (number‑free in chat; numeric context kept in artifacts).
- Thesis snapshot: top‑level `thesis` preserves `action`, `risk_band`, and `readiness` used for headers.
- Per‑timeframe plan snapshot: `plan[tf]` with `entries`/`invalidation`/`targets`, plus `source` (`analysis|fallback`) and `explain` (plain English "Outcome").
- Timescale scores: `timescale_scores` now use `price_change_pct` (renamed from `price_move_pct`) and include `alignment_flag`.
- Polymarket array: top‑level `polymarket[]` when discovery is enabled; artifacts keep probabilities even when chat hides numbers.

See `docs/payload.md` for the complete spec and examples.

### Deterministic consistency gate

Use the gate to verify back‑to‑back universe scans are identical under a safe, deterministic profile. CI can run this after unit tests.

Usage (safe, no sends, no git side effects):
```
TB_DETERMINISTIC=1 TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 \
TB_UNIVERSE_GIT_AUTOCOMMIT=0 TB_UNIVERSE_GIT_PUSH=0 \
python3 scripts/consistency_check.py --config config/universe.yaml --top 10
```

Exit code is non‑zero on drift; logs print the first diff (payload tuple or ranking) to help isolate causes.

***

## Polymarket data path (PPLX‑only)

- Provider: `providers/polymarket_pplx.py` (Perplexity Pro API)
  - Strict JSON array extraction from mixed output
  - Key rotation: `PPLX_API_KEY_1..N`, fallback `PPLX_API_KEY`
  - Enforced model: `sonar`
- Bridge: `scripts/polymarket_bridge.py`
  - Filters by date window, liquidity (optional), keywords (configurable)
  - Caps items, computes internal probability from asset signals (calibrated)
- Chat number gating: hide numbers in TG/Discord via `TB_POLYMARKET_NUMBERS_IN_CHAT=0` (artifacts keep numbers)
- Always‑render empty state if desired: `TB_POLYMARKET_SHOW_EMPTY=1`

***

## Digest delivery (Telegram & Discord)

- Telegram:
  - `scripts/tg_sender.py` respects `TB_HUMAN_DIGEST`/`TB_NO_TELEGRAM`; splits long messages safely
  - Crypto‑only mode: `TB_DIGEST_TELEGRAM_CRYPTO_ONLY=1`
- Discord:
  - `scripts/discord_sender.py` sends chunked embeds; respects `TB_NO_DISCORD`
- Evidence lines in chat are number‑free; artifacts retain numeric metrics
- Confidence display can be toggled for chat:
  - `TB_POLYMARKET_SHOW_CONFIDENCE=0`
  - `TB_SHOW_ASSET_CONFIDENCE_IN_CHAT=0`

Paritised features (Telegram + Discord):
- A+ setups: assets qualifying as high‑quality setups show `[A+ Setup]` in headers and `(A+)` in Quick Summary coin lines.
- Kid‑friendly Quick Summary: end‑of‑digest recap of big picture, leaders, and simple plan.
- Plain‑English phrasing: Executive Take, Weekly Plan, and Engine Thesis are simplified for readability.

***

## Underrated Scanner (utility‑only, latest only)

- Purpose: discover freshly underrated utility crypto projects and alert with evidence.
- Provider: Perplexity API (key rotation); enrichment: CoinGecko market cap + liquidity.
- Filters: exclude memecoins/presales ("meme", "pepe", "inu", "presale", etc.), hype keywords ("viral", "pump"), large‑cap staples (BTC/ETH/XRP/BNB/ADA/DOGE/SOL/etc.), and generic "ecosystem" entries.
- Recency: prompts request `recent_date` and `recent_evidence`; optional hard gate requires items within the last N hours.
- Smart re‑include: previously alerted items are considered again only if they have a new `recent_date` within your window.
- Formatter parity: Telegram and Discord underrated alerts render the same fields and ordering.

Environment variables (.env):
```
# Discovery window and selection
TB_UNDERRATED_PPLX_HOURS=72
TB_UNDERRATED_TOP_N=20

# Dedup/re‑include controls
TB_UNDERRATED_FORCE_ALERTS=0           # one‑off bulk include of top‑N
TB_UNDERRATED_REINCLUDE_RECENT=1       # allow if recent again
TB_UNDERRATED_REQUIRE_RECENT=0         # hard gate on recent_date

# Enrichment threshold and outputs
TB_UNDERRATED_MARKETCAP_THRESHOLD=10000000
TB_UNDERRATED_STORE=data/underrated_store.json
TB_UNDERRATED_OUTDIR=underrated_runs

# Sends and git ops (artifacts/docs only; never .py)
TB_UNDERRATED_ALERT_DISCORD=1
TB_UNDERRATED_ALERT_TELEGRAM=1
TB_UNDERRATED_GIT_AUTOCOMMIT=1
TB_UNDERRATED_GIT_PUSH=1
```

Run examples:
- Safe preview (no sends, no git side effects):
```
TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 \
TB_UNDERRATED_GIT_AUTOCOMMIT=0 TB_UNDERRATED_GIT_PUSH=0 \
python3 scripts/underrated_scanner.py
```

- Strict freshness with sends (e.g., last 72h) and smart re‑include:
```
TB_UNDERRATED_RUN_INTERVAL_DAYS=0 \
TB_UNDERRATED_PPLX_HOURS=72 TB_UNDERRATED_TOP_N=20 \
TB_UNDERRATED_REQUIRE_RECENT=1 TB_UNDERRATED_REINCLUDE_RECENT=1 \
TB_ENABLE_DISCORD=1 TB_UNDERRATED_ALERT_DISCORD=1 \
TB_NO_TELEGRAM=0 TB_UNDERRATED_ALERT_TELEGRAM=1 \
TB_UNDERRATED_GIT_AUTOCOMMIT=1 TB_UNDERRATED_GIT_PUSH=1 \
TB_PPLX_TIMEOUT=45 TB_PPLX_BACKOFF=0.2 \
python3 scripts/underrated_scanner.py
```

Notes:
- Auto‑commit only stages `underrated_runs/` and `data/underrated_store.json` (plus Dev_logs.md when updated). Scripts are never committed.
- Configure dedicated Discord webhook via `DISCORD_UNDERRATED_WEBHOOK_URL` or fall back to existing digest hooks.

***

## Reliability & safety (v4.3)

- Circuit breakers and degraded‑run markers (in progress)
- Retries/backoff on providers/senders (in progress)
- Key rotation for Perplexity
- Strict parsing, schema enforcement, fallbacks

### Hybrid trader robustness gates (opt‑in)
- ML probability gate — blocks BUYs when model probability is below threshold or inference fails (conservative default):
  - Env: `TB_USE_ML_GATE`, `TB_ML_GATE_MODEL_PATH` (default `eval_runs/ml/latest/model.pt`), `TB_ML_FEATURES_PATH`, `TB_ML_GATE_MIN_PROB`.
- ATR volatility filter (15m) — require ATR% within band to avoid dead or chaotic regimes:
  - Env: `TB_USE_ATR_FILTER`, `TB_ATR_LEN`, `TB_ATR_MIN_PCT`, `TB_ATR_MAX_PCT`.
- Higher‑timeframe regime alignment (1h EMA) — trade only in direction of the HTF trend:
  - Env: `TB_USE_HTF_REGIME`, `TB_HTF_EMA_LEN` (e.g., 200).
- Heartbeat notifications — optional liveness pings every N runs when notifications are enabled:
  - Env: `TB_TRADER_NOTIFY_HEARTBEAT`, `TB_HEARTBEAT_EVERY_N`.

- Per‑trade notional cap — hard USD cap per order; quantity is clamped so `qty * entry_price ≤ cap`:
  - Env: `TB_MAX_NOTIONAL_PER_TRADE` (default `1000`).
  - Enforcement points: `calc_position_size()` and a safety‑net before submit in `place_bracket()` within `scripts/hybrid_crypto_trader.py`.

***

## Quick commands recap

- Multi‑asset universe digest (no sends):
```
TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 python3 scripts/tracer_bullet_universe.py --config config/universe.yaml --top 10
```
- Polymarket discovery (PPLX debug):
```
TB_POLYMARKET_DEBUG=1 TB_ENABLE_POLYMARKET=1 python3 scripts/polymarket_bridge.py --max-items 4
```
- Evaluation pipeline:
```
python3 scripts/run_eval_tests.py
python3 scripts/eval_ingest.py --input eval_data/resolved/sample.csv
python3 scripts/eval_runner.py
```

- Standalone Polymarket digest:
```
# Default (PPLX provider). Request 30 items, no client-side cap, write Markdown
TB_POLYMARKET_LIMIT=30 TB_POLYMARKET_PPLX_MAX=0 \
python3 scripts/polymarket_digest.py --format md --output polymarket.md

# Use native public API explicitly (and remove local cap)
python3 scripts/polymarket_digest.py --provider native --full
```
Notes: `--full` applies to the native provider. PPLX count is controlled via `TB_POLYMARKET_LIMIT` and optional `TB_POLYMARKET_PPLX_MAX`.

***

## Polymarket configuration (.env)
- **Objective tape vs narrative cross‑check**
  - Tape: `alpaca.py`, `price.py`, `bars_stock.py`, `timescales.py`
  - Narrative: `perplexity_fetcher.py`, `pplx_fetcher.py`, `narrative_dev.py`
  - Divergence is the explicit gap between story and price across multiple horizons.

- **Semantic relevance gating** — only asset‑relevant content is scored
  - `relevance.py`, `narrative_dev.py`

- **Robust aggregation** — MAD outlier drop + trimmed means
  - `sentiment_utils.py`

- **Decay and timescale alignment** — avoid stale narratives and single‑window bias
  - `narrative_dev.py`, `timescales.py`

- **Confirmation checks (price vs narrative)** — penalize contradictions
  - `confirmation.py`, `timescales.py`

- **Source diversity and de‑duplication**
  - `dedupe_utils.py`, `diversity.py`, `source_weights.py`, `debug_sources.py`

- **Explainability & number‑free chat**
  - `scripts/evidence_lines.py`, `scripts/tg_digest_formatter.py`, `scripts/discord_formatter.py`

- **Continuous evaluation and calibration**
  - `scripts/eval_metrics.py`, `scripts/eval_runner.py`, `scripts/eval_ingest.py`

- **Operational guardrails**
  - `autocommit.py`, `.env.example` (key rotation, push gating, retries/backoff WIP)

***

## Initial data sources

- **Market data (objective)**: Alpaca bars/price
  - `alpaca.py`, `price.py`, `bars_stock.py`
- **News synthesis (structured)**: Perplexity Pro API
  - `perplexity_fetcher.py`, `pplx_fetcher.py`
- **Optional mainstream feed**: CoinDesk RSS
  - `coindesk_rss.py`
- **Prediction markets (reference)**: Polymarket via PPLX
  - `providers/polymarket_pplx.py`, `scripts/polymarket_bridge.py`

Alternative (comma-separated):
```
PPLX_API_KEYS=pk_live_keyA,pk_live_keyB,pk_live_keyC,pk_live_keyD
```

Other toggles and parameters:
```
USE_COINDESK=true
RELEVANCE_THRESHOLD=0.42
LOOKBACK_MINUTES=120
DIVERGENCE_THRESHOLD=1.0
CONFIDENCE_CUTOFF=0.6
NARRATIVE_HALFLIFE_MIN=90
SYMBOL=BTC/USD
ALPACA_API_KEY_ID=
ALPACA_API_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

---

### Polymarket configuration (.env)

Polymarket discovery/mapping can be toggled and tuned via these env vars:

```
# Enable and choose source
TB_ENABLE_POLYMARKET=1
TB_POLYMARKET_SOURCE=pplx   # pplx (Perplexity-backed) recommended; native supported for tests

# Scope and caps
TB_POLYMARKET_ASSETS=BTC,ETH,SOL,XRP
TB_POLYMARKET_MAX_ITEMS=2   # bridge will cap to this; discover_and_map also accepts max_items arg

# Liquidity gating (optional; defaults off)
TB_POLYMARKET_MIN_LIQUIDITY=1000
TB_POLYMARKET_REQUIRE_LIQUIDITY=0  # set 1 to enforce liquidityUSD >= MIN

# Quality gate (if provider supplies quality/score)
TB_POLYMARKET_MIN_QUALITY=0.0

# Time window and activity
TB_POLYMARKET_MIN_WEEKS=0
TB_POLYMARKET_MAX_WEEKS=52
TB_POLYMARKET_MAX_WINDOW_DAYS=30   # additional cap used by bridge
TB_POLYMARKET_TODAY_ACTIVE_ONLY=0  # when 1, show only currently-trading markets

# Misc
TB_POLYMARKET_SHOW_EMPTY=1
TB_POLYMARKET_LIMIT=50
TB_POLYMARKET_TITLE_KEYWORDS=BTC,Bitcoin,ETH,Ethereum,SOL,Solana
TB_POLYMARKET_NUMBERS_IN_CHAT=0
TB_POLYMARKET_DEBUG=0
```

Notes:
- Native provider always enforces a sane end-date window and requires `resolutionSource`.
- PPLX provider relaxes windowing by default; the bridge applies final windowing and caps and honors liquidity gating when enabled.

#### Sender-specific flags
- `TB_POLYMARKET_PPLX_USE_PLAIN_ONLY=1` — force only the dedicated `PPLX_API_KEY` for digest runs (no rotation).
- `TB_POLYMARKET_FALLBACK_NATIVE=1` — allow native fallback when PPLX returns zero items.
- `TB_AUTOCOMMIT_DOCS=1` — auto-commit/push `polymarket_digest.md` after a run (disable with `0` for dry-runs).
- `TB_NO_DISCORD=1` — suppress Discord sends (Telegram already gated by `TB_NO_TELEGRAM`).

## 5) Runbook

Pre-run:
- Ensure Perplexity keys present and PPLX_ENABLED=true
- USE_COINDESK=true recommended
- Start with RELEVANCE_THRESHOLD=0.42

Run:
```
python3 tracer_bullet.py
```

Validate:
- Console shows:
  - Accepted (source, score) list
  - Relevance top-5
  - Decision Preview with adaptive trigger and robust FinBERT kept/dropped
- Artifacts:
  - runs/<id>.json
  - runs/<id>_accepted.txt
  - bars/<id>.csv
- Auto-commit pushes runs/ and bars/

Tuning:
- If accepted < 2 frequently:
  - Set RELEVANCE_THRESHOLD=0.40
  - Use relevance top-5 to calibrate
- If Perplexity empty or 401:
  - Run test_pplx_auth.py and debug_sources.py
  - Check .env keys load in config.py

---

## 6) JSON payload (key fields)

- headlines_count, raw_headlines
- relevance_details.accepted[{headline, relevance, source}], rejected[{headline, relevance}]
- finbert_score, finbert_kept_scores, finbert_dropped_scores
- llm_score, raw_narrative, decayed_narrative
- price_score, volume_z
- divergence, divergence_threshold (adaptive)
- confidence, action, reason
- pplx_provenance (source/title/url list)
- summary, detail

---

## 7) Dev log summary

- Added multi-source ingest: Perplexity Pro API (key rotation), CoinDesk RSS
- Implemented robust FinBERT sentiment (MAD + trimmed mean) on relevant-only
- Enriched BTC topic; tuned relevance gating; added keyword fallback
- Built narrative from accepted-only; conservative confidence fallback
- Added provenance tagging; saved accepted headlines to TXT
- Adaptive divergence trigger based on volume Z
- Debug utilities: inspect_env_pplx, test_pplx_auth, debug_sources
- Improved console output: accepted sources/scores, relevance top-5, decision preview
- Auto-commit of artifacts retained; JSON enriched with robust details and Perplexity provenance

---

## 8) Roadmap milestone update

Milestone: Agent V1 – Stable ingest and robust sentiment

Completed:
- Perplexity rotation + CoinDesk RSS
- Deduplication + semantic gating
- Robust FinBERT (MAD + trimmed mean)
- Narrative from accepted-only; confidence heuristic
- Adaptive trigger (volume-aware)
- Provenance tagging; exports (JSON, CSV, accepted TXT)
- Debug scripts; config hardening

Outstanding:
- Per-source weighting (boost BTC-only; downweight multi-asset wraps)
- Confidence shaping by source diversity and accepted count
- JSON schema versioning; unit tests for drop_outliers, trimmed_mean, adaptive_trigger
- Optional: CoinTelegraph RSS integration
- Optional: Relaxed JSON parsing fallback for Perplexity non-strict responses

---

## 9) Changelog

Agent V1:
- Added: Perplexity Pro API ingestion with multi-key rotation
- Added: CoinDesk RSS ingestion
- Added: Robust FinBERT sentiment (MAD outlier control + trimmed mean)
- Added: Adaptive divergence trigger (volume-aware)
- Added: Provenance tagging and accepted-headlines TXT export
- Improved: Narrative from accepted-only; confidence heuristic
- Improved: Relevance gating with enriched BTC topic and debug visibility
- Fixed: Tokenizer parallelism warnings silenced

---

## Crypto trader (paper) quickstart

- Configure `.env` with Alpaca paper keys and optional Discord webhook.
- Recommended quality/safety flags in `scripts/crypto_signals_trader.py`:
  - `--entry-tolerance-bps 10` (trigger tolerance)
  - `--entry-mid-zone` (midpoint entry when plan has a zone)
  - `--min-rr 2.0` (minimum risk-reward gate based on plan levels)
  - `--cooldown-sec 3600` (1h cooldown)
  - `--order-ttl-min 30` (cancel stale orders)
- Example (single pass):
```
TB_TRADER_OFFLINE=0 TB_NO_TRADE=0 TB_TRADER_NOTIFY=1 TB_ENABLE_DISCORD=1 \
python3 scripts/crypto_signals_trader.py \
  --tf 4h --symbols BTC/USD,ETH/USD \
  --entry-tolerance-bps 10 --entry-mid-zone --min-rr 2.0 \
  --cooldown-sec 3600 --order-ttl-min 30 --debug
```

### 24/7 background (macOS)
- Use a LaunchAgent to run every 5 minutes (no `--loop`); cooldown gates re-entry for 1 hour.
- Logs: `trader_loop.log`, `trader_loop.err` in project root.
- Full plist and commands are in `docs/commands.md` (section: "24/7 background via launchd").
- Tooling: Debug scripts for env inspection and API auth validations

---

## 10) Commit message (squash)

Title:
- agent v1: multi-source ingest (Perplexity rotation + CoinDesk), robust sentiment, adaptive trigger, provenance, exports

Body:
- Add Perplexity Pro API fetcher with key rotation (pplx_fetcher.py), strict JSON parsing, provenance capture
- Add CoinDesk RSS fetcher (coindesk_rss.py) with retry/backoff; merged pre-relevance
- Dedupe merged headlines (dedupe_utils.py), preserve original text
- Relevance pipeline (narrative_dev.py): enriched BTC topic; narrative from accepted-only; confidence heuristic (1→0.55, 2+→0.65)
- FinBERT robust sentiment (finbert.py + sentiment_utils.py): pos-neg scores; MAD outlier drop; 10% trimmed mean; kept/dropped scores exported
- Adaptive divergence trigger (narrative_analysis_extras.py) based on volume Z; reflected in payload and console
- Provenance tagging (provenance.py) and accepted-headlines TXT export (export.py)
- tracer_bullet.py orchestration: multi-source merge, relevance gating, robust sentiment, narrative decay, price/volume, divergence, adaptive action, exports (JSON/CSV/TXT), auto-commit; debug prints for accepted sources and relevance top-5
- Dev tooling: debug_sources.py, test_pplx_auth.py, inspect_env_pplx.py
- Config (config.py): numbered PPLX_API_KEY_1..N and PPLX_API_KEYS fallback; toggles; thresholds; lookbacks

Notes:
- JSON payload includes finbert_kept/dropped counts/scores, pplx_provenance, adaptive divergence threshold
- Legacy sentiment_score retained; tracer uses sentiment_robust

## What's new in V1.1
- Weighted relevance by source (Perplexity > CoinDesk > Alpaca) to improve acceptance precision.
- Perplexity search recency fixed to "day" to curb stale items.
- Alpha-first console output and JSON: signal summary + playbook (no more process-heavy text).
- Telegram integration to deliver the alpha summary to a channel/group.
- DB payload alignment: summary/detail retained; additional fields for audit (relevance_details with raw/weighted and source).

# Tracer Bullet — Crypto News-to-Alpha Agent

Tracer Bullet ingests multi-source headlines, filters for asset relevance, scores narrative vs price/volume, and outputs a trader-facing signal with next steps. Exports artifacts and can DM you via Telegram.

## What’s new (V1.1)
- Per-source weighted relevance (Perplexity > CoinDesk > Alpaca).
- Perplexity “day” recency enforcement and provenance capture.
- Alpha-first console/JSON output (signal + playbook).
- Telegram DM integration for end-of-run push.
- Restored payload summary/detail for DB schema alignment.

## Quick start
1) Configure .env with data/API keys and Telegram (optional).
2) Install deps: `pip3 install -U python-dotenv requests`
3) Run: `python3 tracer_bullet.py` (saves to runs/, bars/, sends Telegram if configured)

## V3 Bias Immunity and Sizing

### Timescale scoring
- Adds short/mid/long horizon analysis using 1-min bars (60/180/360 min tails).
- Each horizon computes:
  - divergence_h = narrative_decayed_proxy − price_score_h
  - price_move_pct_h: abs% change first→last close in horizon
  - volume_z_h: last-bar volume z-score within horizon (0 if n<3)
- Combined divergence:
  - Weighted sum with default weights short=0.50, mid=0.35, long=0.15
  - Weights overrideable via env; renormalized to sum=1.0
- Alignment:
  - alignment_flag True if ≥2 horizons share the combined sign and |divergence_h| ≥ 0.20
- Output:
  - payload.timescale_scores with per-horizon metrics, combined_divergence, aligned_horizons, alignment_flag, weights

### Negative-confirmation weighting
- Small, bounded penalties applied to confidence when confirmations contradict the narrative:
  - price_vs_narrative: notable short-term price move against combined divergence
  - volume_support: low average volume_z for sizable divergence
  - timescale_alignment: lack of alignment for sizable divergence
- Penalties sum and are clamped to a minimum (default -0.05)
- Output:
  - payload.confirmation_checks (name, passed, delta)
  - payload.confirmation_penalty (≤ 0), applied to confidence with floor/cap

### Position sizing (informational)
- Maps final confidence to a target risk size (R) with floors and caps
- Linear scaling between confidence floor and cap to [min_R, max_R]
- Optional volatility normalization hook (off by default)
- Output:
  - payload.position_sizing {confidence, target_R, notes[], params{...}}
- No order placement in V3; Telegram includes a one-liner when target_R > 0

### New environment variables
- Timescales weights:
  - TB_TS_W_SHORT (default 0.50)
  - TB_TS_W_MID (default 0.35)
  - TB_TS_W_LONG (default 0.15)
- Confirmation penalties:
  - TB_CONF_PRICE_VS_NARR (default -0.02)
  - TB_CONF_VOLUME_SUPPORT (default -0.01)
  - TB_CONF_TS_ALIGN (default -0.02)
  - TB_CONF_PENALTY_MIN (default -0.05)
- Sizing:
  - TB_SIZE_CONF_FLOOR (default 0.65)
  - TB_SIZE_CONF_CAP (default 0.85)
  - TB_SIZE_MIN_R (default 0.25)
  - TB_SIZE_MAX_R (default 1.00)

### Example usage
- Override timescale weights:
```
TB_TS_W_SHORT=0.6 TB_TS_W_MID=0.3 TB_TS_W_LONG=0.1 python3 scripts/run.py --symbol BTC/USD --lookback 240 --no-telegram
```
- Tune confirmation penalties:
```
TB_CONF_PRICE_VS_NARR=-0.03 TB_CONF_TS_ALIGN=-0.03 python3 scripts/run.py --symbol BTC/USD --lookback 240 --no-telegram
```
- Adjust sizing floor/caps:
```
TB_SIZE_CONF_FLOOR=0.66 TB_SIZE_MIN_R=0.30 python3 scripts/run.py --symbol BTC/USD --lookback 240 --no-telegram
```

## V3.1 Multi-Asset Universe

### Universe scanning
- Multi-asset analysis across crypto and stocks
- Market hours awareness (RTH/CLOSED for stocks, 24x7 for crypto)
- Configurable universe via `config/universe.yaml`
- Top-N ranking and digest generation

### Usage
- Scan universe:
```
TB_NO_TELEGRAM=1 TB_ALLOW_STUB_BARS=1 python3 scripts/scan_universe.py --config config/universe.yaml --top 5 --debug
```
- Override symbols:
```
python3 scripts/scan_universe.py --symbols "BTC/USD,AAPL,MSFT" --top 3
```
- Custom config:
```
python3 scripts/scan_universe.py --config my_universe.yaml --top 10
```

### Auto-commit and mirroring
- Mirror universe results to runs/:
```
TB_UNIVERSE_MIRROR_TO_RUNS=1 python3 scripts/scan_universe.py --config config/universe.yaml --top 5
```
- Auto-commit universe results:
```
TB_UNIVERSE_GIT_AUTOCOMMIT=1 python3 scripts/scan_universe.py --config config/universe.yaml --top 5
```
- Auto-commit and push:
```
TB_UNIVERSE_GIT_AUTOCOMMIT=1 TB_UNIVERSE_GIT_PUSH=1 python3 scripts/scan_universe.py --config config/universe.yaml --top 5
```

---

## Ops & Automation — Hybrid Trader (24/7)

This project includes a fully automated hybrid crypto trader with self-healing and monitoring.

- **Start loop**: `scripts/start_hybrid_loop.sh`
  - Loads `.env`, sets conservative defaults, starts a periodic ML retrainer and the trader loop.
  - Preflight: if `config/promoted_params.json` is missing/stale (> `TB_START_MAX_PROMOTED_AGE_DAYS`, default 8), it auto-runs `scripts/weekly_propose_canary.sh` before starting.
  - Artifacts auto-commit/push (JSON/CSV/MD only; never `*.py`) controlled by `TB_AUTOCOMMIT_ARTIFACTS`, `TB_AUTOCOMMIT_PUSH`.

- **Robustness gates (env toggles)**
  - `TB_USE_ML_GATE`, `TB_ML_GATE_MODEL_PATH`, `TB_ML_GATE_MIN_PROB`
  - `TB_USE_ATR_FILTER`, `TB_ATR_LEN`, `TB_ATR_MIN_PCT`, `TB_ATR_MAX_PCT`
  - `TB_USE_HTF_REGIME`, `TB_HTF_EMA_LEN`

- **Notifications**
  - Enable via `TB_TRADER_NOTIFY=1` and channels: Discord (`TB_ENABLE_DISCORD=1`, `DISCORD_WEBHOOK_URL`), Telegram allowed unless `TB_NO_TELEGRAM=1`.
  - Heartbeats every N runs: `TB_TRADER_NOTIFY_HEARTBEAT=1`, `TB_HEARTBEAT_EVERY_N`.

- **Watchdog (cron)**
  - Every 2 minutes restarts the loop if it dies: `scripts/watchdog_hybrid.sh`  (crontab tag `# com.tracer.watchdog-hybrid`).

- **Daily health check (cron 09:00)**
  - `scripts/health_check.sh` verifies process, log freshness, recent `runs/`, and promoted params freshness.
  - Self-heal: if `promoted_params.json` is stale/missing, it runs `scripts/weekly_propose_canary.sh` once (lock-protected), then re-checks before alerting.
  - Alerts only on failure (Discord/Telegram gated by env).

- **Weekly propose+canary**
  - Primary cron Sundays 03:00; backup Wednesdays 03:00 to ensure refresh.

- **Quick ops**
  - Start live: `bash scripts/start_hybrid_loop.sh`
  - Tail logs: `tail -n 200 -f trader_loop.err` and `tail -n 200 -f trader_loop.log`
  - Processes: `ps ax | egrep 'hybrid_crypto_trader.py|ml_retrainer.py' | egrep -v egrep`

See `docs/commands.md` for concrete cron lines and stop/disable examples.
