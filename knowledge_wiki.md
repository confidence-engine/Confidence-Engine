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

# File-by-file summaries

- tracer_bullet.py
  - Orchestrates full run: fetches headlines (Alpaca + Perplexity rotation + CoinDesk if enabled), dedupes, semantic relevance gating, robust FinBERT sentiment on relevant-only headlines, narrative built from accepted headlines with confidence heuristic, narrative decay, price/volume scores, divergence and action with adaptive trigger, exports (JSON, bars CSV, accepted TXT), auto-commit, console preview.
  - Key features: robust sentiment aggregation (MAD outlier drop + trimmed mean), adaptive divergence trigger, provenance tagging, conservative confidence fallback when narrative missing but accepted exists.

- finbert.py
  - FinBERT sentiment scoring.
  - sentiment_score: legacy mean.
  - sentiment_robust: per-headline pos-neg scores in [-1,1], MAD outlier drop, 10% trimmed mean. Returns (aggregate, kept_scores, dropped_outliers).

- sentiment_utils.py
  - Robust stats helpers: trimmed_mean, mad, drop_outliers (MAD z-score).

- narrative_dev.py
  - Relevance gating and narrative construction.
  - BTC_TOPIC enriched for BTC.
  - filter_relevant: accepted/rejected with scores.
  - make_from_headlines: narrative from accepted-only; confidence 0.55 (1 headline), 0.65 (2+); neutral momentum (blend with FinBERT downstream).

- pplx_fetcher.py
  - Perplexity Pro API client (sonar-pro chat completions).
  - fetch_pplx_headlines_with_rotation: rotates multiple API keys; returns (titles, items, err). Strict JSON response parsing.

- coindesk_rss.py
  - CoinDesk RSS fetch with retry/backoff; returns titles.

- dedupe_utils.py
  - Normalize + dedupe titles; preserves original text.

- provenance.py
  - Tag origins for accepted headlines: perplexity, alpaca, coindesk, unknown.

- narrative_analysis_extras.py
  - adaptive_trigger: adjust divergence trigger by volume Z; clamp to [0.6, 1.5].

- export.py
  - export_run_json: runs/<id>.json
  - save_bars_csv: bars/<id>.csv
  - save_accepted_txt: runs/<id>_accepted.txt with [source] relevance | headline

- config.py
  - .env loader and settings.
  - Supports PPLX_API_KEY_1..N or PPLX_API_KEYS (comma) or single PPLX_API_KEY.
  - Toggles: USE_COINDESK, PPLX_ENABLED; thresholds, lookbacks.

- debug_sources.py
  - Dev utility: print counts/samples from Alpaca, Perplexity (rotation), CoinDesk.

- test_pplx_auth.py
  - Dev utility: 200/401 checks for each Perplexity key.

- inspect_env_pplx.py
  - Dev utility: raw env string inspection for PPLX_API_KEYS.
-----------------
# How it works

## Data ingestion
- Alpaca latest_headlines(symbol, limit)
- Perplexity Pro API via sonar-pro with key rotation (PPLX_API_KEY_1..N or PPLX_API_KEYS)
- CoinDesk RSS (USE_COINDESK toggle)
- Deduplication across sources

## Relevance gating
- Semantic similarity to enriched BTC topic
- RELEVANCE_THRESHOLD 0.40–0.45
- Keyword fallback if 0 accepted

## Sentiment + narrative
- FinBERT robust sentiment (relevant-only): MAD outlier drop + 10% trimmed mean
- Narrative from accepted headlines; confidence by count; composite signal blends LLM (neutral baseline) and FinBERT; decay by staleness

## Price/volume and divergence
- Price score from bars; Volume Z
- Divergence = decayed narrative − price score
- Adaptive trigger based on volume (lower when high participation, higher when low)

## Decision and outputs
- BUY/HOLD with confidence cutoff, divergence trigger, volume floor
- Exports JSON, accepted TXT, bars CSV; auto-commit
- Console preview: provenance, top-5 relevance (debug), beginner-friendly summary

--------------------------

# Runbook – daily use

## Pre-run
- Ensure .env has Perplexity keys (PPLX_API_KEY_1..N or PPLX_API_KEYS) and PPLX_ENABLED=true
- USE_COINDESK=true recommended
- Start with RELEVANCE_THRESHOLD=0.42

## Run
- python3 tracer_bullet.py

## Validate
- Console shows:
  - Accepted (source, score) with ~3–8 items
  - Decision Preview with adaptive trigger and robust FinBERT kept/dropped
- Artifacts:
  - runs/<id>.json
  - runs/<id>_accepted.txt
  - bars/<id>.csv
- Auto-commit pushes runs/ and bars/

## Tuning
- If accepted < 2 repeatedly:
  - Set RELEVANCE_THRESHOLD=0.40
  - Check [Relevance top-5] output
- Verify Perplexity returns titles (debug_sources.py) and keys 200 (test_pplx_auth.py)

## Per-source weighting and Perplexity recency

- Weighted relevance:
  - Each headline’s semantic relevance is multiplied by a source weight:
    - Perplexity: 1.10
    - CoinDesk: 1.05
    - Alpaca: 1.00
  - JSON payload -> relevance_details.accepted[].raw_relevance and weighted_relevance
- Perplexity recency:
  - web_search_options.search_recency_filter = "day" (past 24h)
  - Ensures fresher results and reduces stale headlines

