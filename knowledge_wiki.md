# Roadmap (Milestone-Based) — Project Tracer Bullet

A living, milestone-driven plan from V1 tracer bullet to small live capital, emphasizing reliability, explainability, and leak-free validation.

Last updated: 2025-08-08.

## Guiding Objectives

- Ship an end-to-end tracer quickly, then harden.
- Keep features interpretable; earn complexity with evidence.
- Capture rich evidence per signal to build our proprietary dataset.
- Validate via event-ordered, leak-free backtests before scaling.

***

## V1 — Tracer Bullet (single-asset, end-to-end)

Goal: One clean loop from ingest → analyze → decide → log → notify, with optional paper order.

Scope:
- Single symbol: BTCUSD.
- Data: Alpaca bars(last 60–120m) + latest headlines.
- Narrative: dev adapter returns strict JSON; schema-validated; raw JSON stored.
- NLP: FinBERT headline sentiment blended with LLM polarity; exponential time decay.
- TA: RSI(14), MA(10 vs. 50) slope, MACD histogram, volume z-score → Price Score.
- Divergence: z(Narrative) − z(Price); thresholds gate entries (long-only).
- Execution: Alpaca paper order (market), idempotent client order IDs.
- Logging: SQLite signals, trades, prompts; store raw narrative JSON.
- Notifications: Telegram “style analysis” message per signal (human-readable rationale).

Acceptance criteria:
- End-to-end run produces at least one decision preview and, when enabled, a paper trade with full context logged.
- Narrative JSON parser rejects malformed payloads safely (no trade on invalid parse).
- Telegram messages include summary, scores, divergence, TA context, decision, and trace IDs.

Out-of-scope:
- Shorts; novelty weighting; backtesting harness.

Why now:
- Establishes the thin thread; surfaces integration risks immediately.

***

## V2 — Better Signals (event triggers, decay, explainability)

Goal: Reduce noise and improve precision; expand to a curated multi-asset set.

Scope:
- Event-driven scans: trigger focused scans on headline deltas (novelty/first-seen shifts).
- Narrative decay active (configurable half-life).
- Basic novelty weighting: de-duplicate echoes; boost first-source items.
- Universe: 10–20 narrative-sensitive symbols (news-dense equities + selective crypto).
- Explainability+: enrich Telegram + DB with anchor headlines, entities, and reason codes.

Acceptance criteria:
- Fewer, higher-quality signals per day; explainability artifacts stored for each signal.
- No duplicate triggers for the same story within a cooldown window.
- Stable event loop under typical data loads.

Out-of-scope:
- Backtesting; parameter governance.

Why now:
- Aligns signal timing with information flow; improves trust and review efficiency.

***

## V3 — Backtesting & Parameter Governance (leak-free)

Goal: Prove the edge with event-ordered replay and set disciplined parameter update cadence.

Scope:
- Event-ordered backtester: replay timestamped bars + headlines; compute features point-in-time only; evaluate forward returns (e.g., 15m/60m).
- Walk-forward validation: rolling windows for tuning/validation; prevent regime overfit.
- Parameter governance: small set of tunables (divergence T, confidence C, decay half-life H, TA weights) with monthly review/update cadence.
- Cohort analytics: performance by asset, time-of-day, volatility regime, event type.

Acceptance criteria:
- Reproducible backtest runs with documented in-sample/out-of-sample splits.
- Published thresholds from out-of-sample results.
- Cohort report identifying where edge concentrates.

Out-of-scope:
- Execution microstructure tweaks; live capital.

Why now:
- Converts plausible signals into evidence-backed settings; guards against hidden look-ahead.

***

## V4 — Execution Quality & Microstructure

Goal: Improve realized PnL by reducing slippage and avoiding thin/hostile microstructure.

Scope:
- Order tactics: prefer market in normal spreads; switch to limit or small slices when spreads widen or volatility spikes.
- Position sizing: scale notional by volatility and confidence; enforce max per-symbol and total risk caps.
- Cooldowns: stricter symbol/story-level cooldowns to avoid clustering.

Acceptance criteria:
- Measurable reduction in slippage vs. V3 baseline on paper.
- Fewer clustered entries around the same narrative burst.
- No increase in failure-to-fill beyond acceptable bounds.

Out-of-scope:
- Options overlays; streaming infra.

Why now:
- Good signals can still lose with poor fills; execution quality compounds returns.

***

## V5 — Data Breadth & Explainability+

Goal: Strengthen narrative signal; make audit and postmortems faster.

Scope:
- Optional crowd/attention proxy: carefully filtered social signals as secondary evidence (never primary gate).
- Source credibility learning: adapt weights for sources with consistent predictive value.
- Case file per signal: compact artifact bundling summary, quotes, entities, scores, divergence, and outcome link.

Acceptance criteria:
- Higher precision without a surge in false positives.
- Each trade explainable via a single, human-readable case file.

Out-of-scope:
- Automated model selection; deep learning expansions.

Why now:
- Layered evidence improves robustness and speeds iteration.

***

## V6 — Small Live Capital, Monitoring, and Safety

Goal: Move beyond paper with strict guardrails and observability.

Scope:
- Live deployment (small size): daily loss limits, per-trade caps, and circuit breakers (halt on repeated failures).
- Monitoring: hourly heartbeats, daily summaries, anomaly alerts (e.g., sudden silence in a usually noisy symbol).
- Versioning: tag every signal/trade with code/prompt/params version.

Acceptance criteria:
- Stable live operation for a defined trial period (e.g., 4–6 weeks) with automatic rollback path.
- Full reproducibility of any live decision.

Out-of-scope:
- Aggressive scaling; options strategies.

Why now:
- Real-world frictions manifest only with live orders; start small and safe.

***

## V7 — Regime Adaptation & Selective Automation

Goal: Scale where edge persists; adapt weights by simple regimes.

Scope:
- Regime-aware reweighting: small, interpretable set of regimes (e.g., volatility buckets, time-of-day) that adjust TA/narrative weights or thresholds.
- Selective automation: fully automate cohorts with robust edge; keep others alert-only or at reduced size.
- Quarterly research loop: refresh asset list, revalidate parameters, and retire decayed cohorts.

Acceptance criteria:
- Consistent performance across regimes with limited parameter drift.
- Documented decisions to scale or de-scale specific cohorts.

Out-of-scope:
- Ultra-low-latency infra unless justified by measured benefit.

Why now:
- Sustain performance as conditions change; scale responsibly.

***

## Cross-Cutting Workstreams

- Reliability & Resilience (all versions)
  - Strict JSON schemas; repair-then-reject for narrative; never trade on invalid parses.
  - Perplexity key rotation; retries with exponential backoff; timeouts; circuit breakers.
  - TA-only degradation mode when narrative is unavailable; safe halts on data failures.
  - APScheduler-based orchestration for interval/cron/event jobs[1][2][3].

- Documentation & Evidence (continuous)
  - Wiki: architecture, data contracts, prompts, test matrix, runbooks, decisions.
  - Dev log: daily entries of changes, issues, next steps.
  - Version tags baked into signals/trades for reproducibility.

- Compliance & Safety (V6+)
  - Environment separation; secrets in .env; logging hygiene; clear rollback procedures.

***

## Milestone Checklists

V1 Checklist:
- Alpaca connectivity (account/clock), bars+headlines fetch stable[4][5][6].
- Narrative dev adapter returns valid JSON; parser tests pass.
- TA indicators deterministic on fixtures.
- Divergence computation and thresholds wired; preview decision prints.
- Optional: paper trade placed; signals/trades/prompts stored; Telegram message received.

V2 Checklist:
- Event trigger pipeline stable (dedupe echoes).
- Decay/novelty parameters configurable.
- Multi-asset scanning within latency goals.
- Telegram includes anchor quotes/entities; reason codes standardized.

V3 Checklist:
- Event-ordered backtester produces forward returns without leaks.
- Walk-forward results documented; thresholds updated via governance.
- Cohort analysis report generated and reviewed.

V4 Checklist:
- Execution tactics implemented with toggles.
- Slippage metrics show improvement; no excessive unfilled orders.

V5 Checklist:
- Secondary attention proxy integrated behind primary gates.
- Source weighting updated on cadence; case files generated per signal.

V6 Checklist:
- Live trial deployed with strict caps; monitoring + alerts active.
- Circuit breaker tested; rollback procedure rehearsed.

V7 Checklist:
- Regime definitions documented; weights vary by regime with evidence.
- Automation level per cohort justified by stable metrics.

***

## Risks and Mitigations (snapshot)

- Narrative parser fragility → strict schema, repair-then-reject, curated test corpus.
- API rate limits/outages → key rotation, backoff, TA-only mode, circuit breakers.
- Look-ahead bias → event-ordered replay, lagged features, walk-forward validation.
- Overfitting/indicator bloat → small, interpretable feature set; robustness checks; governance cadence.
- Execution slippage → microstructure-aware order rules; measure impact before adding complexity.

***

## Operating Cadence

- Daily: run agent; update dev log at EOD; triage any errors; light wiki edits for new learnings.
- Weekly: review signals and cohort metrics; adjust backlog; curate wiki Decisions.
- Monthly: parameter governance meeting; walk-forward update; asset universe refresh.
- Quarterly: regime analysis; scale/retire cohorts; research new sources or tactics.

***

## Appendices

- References for tooling patterns:
  - APScheduler overview and scheduling patterns[1][2][3].
  - Alpaca paper trading/Trading API basics for connectivity and orders[4][5][6][7][8].

Notes:
- Templates referenced are for visualization inspiration only; our roadmap remains code-first and evidence-driven[9][10][11][12][13][14][15][16].

Sources
[1] An Introduction to APScheduler: Task Scheduling in Python https://www.devtooler.com/an-introduction-to-apscheduler-task-scheduling-in-python/
[2] Job Scheduling in Python with APScheduler | Better Stack Community https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/
[3] APScheduler - PyPI https://pypi.org/project/APScheduler/
[4] How to Start Paper Trading with Alpaca's Trading API https://alpaca.markets/learn/start-paper-trading
[5] Paper Trading - Alpaca API Docs https://docs.alpaca.markets/docs/paper-trading
[6] About Trading API - Alpaca API Docs https://docs.alpaca.markets/docs/trading-api
[7] The Ultimate Alpaca Markets Paper Trading Guide in Python https://wire.insiderfinance.io/alpaca-algorithmic-trading-api-in-python-part-1-getting-started-with-paper-trading-efbff8992836
[8] Alpaca Trading API Guide - A Step-by-step Guide - Algo Trading 101 https://algotrading101.com/learn/alpaca-trading-api-guide/
[9] Free Project Roadmap Templates | Smartsheet https://www.smartsheet.com/content/project-roadmap-templates
[10] Project Roadmap Template | Jira Templates - Atlassian https://www.atlassian.com/software/jira/templates/project-roadmap
[11] 45+ Free Roadmap Templates & Examples - Miro https://miro.com/templates/roadmap/
[12] Software Roadmap Templates - Roadmunk https://roadmunk.com/roadmap-templates/software-roadmap/
[13] Free Roadmap Templates - Office Timeline https://www.officetimeline.com/roadmaps/templates
[14] 179+ Roadmap PowerPoint Templates & Presentations Slides https://slidemodel.com/templates/tag/roadmap/
[15] Best Project Roadmap Templates from Notion | Notion Marketplace https://www.notion.com/templates/category/project-roadmap
[16] 5 Roadmap Templates - Project Manager https://www.projectmanager.com/blog/roadmap-templates
[17] Free and customizable roadmap templates - Canva https://www.canva.com/templates/s/roadmap/
[18] Customizable project timeline and roadmap templates https://create.microsoft.com/en-us/templates/timelines
[19] 4.1 Advanced Python Scheduler - Jodoo https://help.jodoo.com/en/articles/9992509-4-1-advanced-python-scheduler
[20] Introduction to APScheduler - Better Programming https://betterprogramming.pub/introduction-to-apscheduler-86337f3bb4a6

----------------

Great—locking in Option 1. I’ll deliver the “everything-first” bundle now: final file map, data contracts, thresholds, module responsibilities, and paste-ready wiki stubs. Once you skim and say “Start,” I’ll move into step-by-step implementation in small, runnable chunks.

Below is your complete bundle to review and paste into the repo/wiki.

# 1) Final File Map (V1.0 → V1.4)

repo/
- README.md (already updated)
- roadmap.md (milestones + acceptance)
- dev_log.md (daily; update EOD)
- wiki/ (GitHub Wiki or docs/ if preferred)
  - Architecture.md
  - Data-Contracts.md
  - Prompt-Engineering.md
  - Analysis-Design.md
  - Runbooks.md
  - Testing-TDD.md
  - Decisions.md
  - Roadmap.md
- .env.example
- requirements.txt
- tracer_bullet.py
- agent.py (V1.2+)
- src/
  - config.py
  - adapters/
    - narrative_base.py
    - narrative_local.py
    - narrative_perplexity.py (V1.4 stub; active in V2)
  - data/
    - alpaca_fetcher.py
    - yfinance_fetcher.py (V3)
  - nlp/
    - sentiment_finbert.py
    - keyword_extractor.py (V2)
  - analysis/
    - price_momentum.py
    - narrative_momentum.py
    - divergence.py
  - execution/
    - trade_executor.py
    - risk.py
  - db/
    - db_manager.py
    - schema.sql
  - notifier/
    - telegram_notifier.py
  - utils/
    - logging_setup.py
    - retry.py
    - time_utils.py

# 2) .env.example (paste-ready)

ALPACA_API_KEY_ID=
ALPACA_API_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
ENV=dev
PROMPT_VERSION=prompt_v0.1
VERSION_TAG=v1.0.0
SYMBOL=BTCUSD
LOOKBACK_MINUTES=120
HEADLINES_LIMIT=10
DIVERGENCE_THRESHOLD=1.0
CONFIDENCE_CUTOFF=0.6
COOLDOWN_MINUTES=15
NARRATIVE_HALFLIFE_MIN=90

# 3) requirements.txt (V1-ready; paste-ready)

pandas
numpy
requests
python-dotenv
pydantic
alpaca-trade-api
transformers
torch
spacy
python-telegram-bot
APScheduler

# 4) Data Contracts (paste into Wiki/Data-Contracts.md)

Narrative JSON (v0.1)
- narrative_summary: string (<=512 chars)
- narrative_momentum_score: float in [-1, +1]
- confidence: float in 
- salient_entities: list[str] (<=5)
- anchor_quotes: list[str] (<=3)
- metadata (optional):
  - source_sample: list[str] (<=5)
  - model_name: string
  - created_at: ISO-8601

Validation policy
- Enforce ranges and required keys.
- One “repair” attempt (fix common JSON issues).
- If still invalid → drop and log; never trade on invalid parses.

SQLite schema (initial)
- signals(id, ts, symbol, narrative_score, price_score, divergence, confidence, action, reason_code, json_blob, version_tag)
- trades(id, signal_id, ts, side, qty, fill_price, status, pnl_15m, pnl_60m)
- prompts(id, ts, symbol, prompt_hash, model_name, tokens_in, tokens_out, latency_ms)

Telegram message spec (style analysis)
- Title: [Signal] {SYMBOL} — {Bias/Strength}
- Story: {summary}. Confidence: {confidence}
- Narrative vs. Tape: Narrative {narr}, Price {price} → Divergence {div}
- TA: RSI {rsi}, MA(10>50) {up/down}, MACD hist {+/−}, Vol Z {z}
- Decision: {buy/sell/hold} {qty} (paper). Reason: {reason}
- Trace: signal_id={id}, version={version_tag}

# 5) Prompt Template (paste into Wiki/Prompt-Engineering.md)

Prompt header (prompt_v0.1)
- Respond with JSON only. No prose, no markdown. Use keys: narrative_summary (string, <=512 chars), narrative_momentum_score (float in [-1,1]), confidence (float in ), salient_entities (list of <=5 strings), anchor_quotes (list of <=3 short strings), metadata (object with optional source_sample (list[str], <=5), model_name (string), created_at (ISO-8601)).
- If uncertain, lower confidence rather than guessing.
- Escape quotes, ensure valid JSON, and include all keys even if empty lists.

Failure/repair notes
- Typical errors: trailing commas, unescaped quotes, missing keys, wrong types.
- Parser will attempt one repair; otherwise reject.

# 6) Analysis Design (paste into Wiki/Analysis-Design.md)

Narrative Score (v0.1)
- Linear blend: Score = 0.6*LLM_polarity + 0.4*FinBERT_sentiment
- Time decay: exponential with half-life H=90min applied since last headline timestamp
- Confidence: from LLM, clamped 

Price Score (v0.1)
- Indicators:
  - RSI(14) → scaled to [-1, +1]
  - MA slope: short(10) vs medium(50), normalized
  - MACD histogram: sign/strength normalized
  - Volume z-score: credibility multiplier
- Combine: weighted average (start equal weights) × credibility from volume

Divergence & Triggers (initial)
- divergence = z(narrative_score) − z(price_score)
- Entry (V1, long-only):
  - |divergence| > 1.0
  - confidence > 0.6
  - volume z-score > −0.5
  - cooldown >= 15m per symbol
- Reason codes: NARR_LEADS_PRICE, LOW_CONFIDENCE, WEAK_VOLUME, COOLDOWN_ACTIVE, INVALID_PARSE

# 7) Roadmap acceptance slices for V1 (paste into roadmap.md if missing)

V1.0 — Preview (no orders)
- One symbol (BTCUSD), fetch bars+headlines, narrative JSON (dev), compute TA, divergence, preview decision, Telegram message.
- Acceptance: clean preview with valid JSON → decision and message.

V1.1 — Paper orders
- Enable Alpaca paper trade; cooldown; max positions; persist to SQLite.
- Acceptance: at least one paper trade with full context stored and linked to alert.

V1.2 — Scheduler & resilience
- APScheduler run loop; retries/backoff; TA-only degrade; circuit breaker for narrative failures.
- Acceptance: unattended run for hours without crashes; alerts on failures.

V1.3 — Event-lite & explainability
- Headline-delta trigger; reason codes standardized; refined message format.
- Acceptance: fewer, better signals; audit-grade messages.

V1.4 — Hardening
- Unit tests (parser, indicators, divergence, DB IO); config-driven thresholds; version_tag in signals.
- Acceptance: tests pass; reproducible runs.

# 8) ADRs (paste into Wiki/Decisions.md)

ADR-001 — First Symbol
- Decision: BTCUSD (24/7 flow)
- Alternatives: TSLA (market hours)
- Consequences: continuous loop testing, more micro-noise

ADR-002 — Tracer Feature Set
- Decision: Option B (LLM+FinBERT, RSI/MA/MACD/vol z)
- Rationale: better precision without delaying tracer

ADR-003 — Narrative Scoring
- Decision: Linear blend + exponential decay (H=90min)
- Rationale: transparent and testable first pass

ADR-004 — Initial Thresholds
- Decision: |div|>1.0, conf>0.6, vol_z>−0.5, cooldown=15m, long-only

ADR-005 — Notifications
- Decision: Telegram “style analysis” via python-telegram-bot

# 9) Runbooks (paste into Wiki/Runbooks.md)

Telegram setup
- Create bot with @BotFather; get token.
- Start chat with bot; get chat_id via getUpdates or helper snippet.
- Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env.
- Test send_message.

Perplexity key rotation (V2+)
- PERPLEXITY_KEYS=key1,key2,key3 in .env
- Round-robin on 429/5xx; exponential backoff; alert after N failures.

Alpaca paper sanity checks
- Verify account endpoint, clock status, and test a tiny market order/cancel in paper.

Failure recovery
- Parser invalid: skip trade, log sample, add test case.
- Narrative down: TA-only mode (reduced risk) or pause entries.
- Market data down: pause trading; send alert.

# 10) Test Plan (paste into Wiki/Testing-TDD.md)

Parser tests
- Accept valid payloads; repair minor JSON errors; reject missing keys/range violations; no-trade on invalid.

Indicator tests
- RSI/MA/MACD fixed fixtures with expected outputs.
- Volume z-score deterministic on sample series.

Divergence logic
- Synthetic cases around thresholds; cooldown enforcement.

DB integrity
- Schema boot; insert/retrieve; idempotent inserts by signal_id.

Backtesting guardrails (V3+)
- Event-ordered replay; lagged features; walk-forward splits; cohort metrics.

# 11) Initial thresholds and params (final confirmation)

- DIVERGENCE_THRESHOLD=1.0
- CONFIDENCE_CUTOFF=0.6
- COOLDOWN_MINUTES=15
- NARRATIVE_HALFLIFE_MIN=90
- Long-only in V1

# 12) Immediate next step after this bundle

Once this is reviewed, say “Start,” and I’ll begin implementation with V1.0 in small, runnable chunks in this order:
1) requirements.txt + .env.example + src/config.py
2) src/data/alpaca_fetcher.py (bars+headlines) + sanity check
3) src/adapters/narrative_local.py + Pydantic model + parser tests (fixtures)
4) src/nlp/sentiment_finbert.py (FinBERT scoring)
5) src/analysis/price_momentum.py + src/analysis/narrative_momentum.py + src/analysis/divergence.py
6) tracer_bullet.py orchestration (preview mode); console + Telegram message
7) src/db/schema.sql + src/db/db_manager.py (wire after preview looks sane)
8) src/execution/trade_executor.py + src/execution/risk.py (V1.1)
9) agent.py + APScheduler (V1.2)

