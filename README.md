Project Tracer Bullet
Here is a draft for the README introduction that puts the Tracer Bullet philosophy and mission front and center, clearly distinguishing it from generic sentiment bots. This will ensure anyone reading knows the core vision and agile process powering your divergence engine:

***

# Tracer Bullet — The Agile Divergence Engine for Crypto Alpha

## Our Mission: Exploit Emotional-Math Gaps with Perfect Discipline

Financial markets are driven by human emotion as much as by fundamentals. Crowd fear, greed, and narrative bias often cause prices to diverge significantly from their true, statistical values — creating opportunities for disciplined, rational traders.

**Tracer Bullet is not just another sentiment bot. It is a purpose-built divergence engine, designed to detect the market’s “lie” between story and price, and deliver actionable alpha signals in real time.**

We ask the key question nonstop:

> *"The news narrative is strongly positive (or negative), but why hasn’t the price moved yet? Is the market underreacting or overreacting?"*

By measuring and quantifying these divergences rigorously, filtering for relevance and confidence, and adapting thresholds dynamically, Tracer Bullet provides **real-time, explainable**, and **action-ready** trading signals — all delivered with evidence and playbook guidance directly to your Telegram DM.

## Why Tracer Bullet Is Different

- **Philosophy-first:** We shipped a fast, end‑to‑end prototype (V1) to validate our core edge early, avoiding premature complexity.
- **Elegant simplicity:** We leverage two clean, high-quality signal pillars — Alpaca price data and Perplexity synthesized narrative — before adding noisy social feeds.
- **Hybrid intelligence:** Combining quantitative “oracle” data and qualitative “psychologist” narrative to find true market inefficiencies.
- **Divergence over sentiment:** Unlike traditional news sentiment bots, we trade the **gap** between narrative and price, where predictable emotional mistakes happen.
- **Adaptive and auditable:** Confidence shaping, source weighting, catalyst tagging, and full payload provenance ensure discipline and continuous improvement.

## How We Work

- Ingest multi-source crypto news with semantic filtering for asset relevance.
- Score narrative sentiment versus price/volume context.
- Compute divergence signal with adaptive thresholds.
- Generate concise alpha-first summaries and next steps.
- Deliver signals and evidence automatically via Telegram DMs.
- Persist rich data for audit, backtesting, and iterative tuning.

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

## 7) Dev log summary (2025-08-09)

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
