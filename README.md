Project Tracer Bullet

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
- Choose the license that matches your goals. If unsure, start with MIT for flexibility.


