# Roadmap (Milestone-Based) — Project Tracer Bullet

A living, milestone-driven plan from V1 tracer bullet to small live capital, emphasizing reliability, explainability, and leak-free validation.

Last updated: 2025-08-20.

> Prefer a concise, structured version? See [Roadmap_CLEAN.md](Roadmap_CLEAN.md).

## Guiding Objectives

- Ship an end-to-end tracer quickly, then harden.
- Keep features interpretable; earn complexity with evidence.
- Capture rich evidence per signal to build our proprietary dataset.
- Validate via event-ordered, leak-free backtests before scaling.

***

## Status Summary (as of 2025-08-20)

- V1 — Tracer Bullet: DONE
- V2 — Crowd Immunity: DONE
- V3 — Bias Immunity + Sizing: DONE
- V3.1 — Multi-Asset Foundations: DONE
- V3.3 — Evidence Lines + Polymarket (read-only): DONE
- V3.4 — Evaluation Pipeline: NOT DONE
- V4.2 — Backtesting & Governance: NOT DONE
- V4.3 — Reliability Ops Hardening: IN PROGRESS (nightly hit‑rate self‑checks wired, broadened non‑.py auto‑commit)
- V5 — 24/7 Cloud Agent: NOT DONE
- V6 — Paper Execution & Risk: NOT DONE
- V7 — Live Capital: NOT DONE

## Now → Near → Next

- Now — v4.3 Reliability Ops Hardening
  - Nightly self‑checks: asset hit‑rate trend append + regression compare, with env tunables
  - Auto‑commit scope broadened (stage all; unstage *.py) to ensure `runs/*.json` and `universe_runs/metrics.csv` are pushed
  - Unified self‑checks: schema/digest validations, degraded‑run markers, circuit breakers
  - Acceptance: 3‑day burn‑in; <1% degraded runs; zero crashes

- Near — v3.4 Evaluation Pipeline
  - Outcome capture: event-ordered snapshots; resolved outcomes archived to CSV/JSON
  - Metrics: Brier/log-loss, calibration curves; cohort win-rates; odds convergence
  - Reproducibility: ≥50 resolved obs; deterministic scripts

- Next — v4.2 Backtesting & Governance
  - Event-ordered replay of bars + headlines; point-in-time features only
  - Walk-forward validation; rolling IS/OOS; publish selected thresholds
  - Cohort analytics; monthly governance cadence

- Later
  - v5 24/7 Cloud Agent (GH Actions cadence, secrets mgmt, monitored rollback)
  - v6 Paper Execution & Risk (dry-run fills, portfolio caps, kill-switches)
  - v7 Live Capital (small guarded trial with strict loss/risk controls)

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


---------------------

# Milestone: Agent V1 – Stable ingest and robust sentiment

## Completed
- Perplexity integration with key rotation
- CoinDesk RSS ingest
- Deduplication and semantic relevance gating
- Robust FinBERT (MAD + trimmed mean) on relevant-only
- Narrative from accepted-only; confidence heuristic
- Adaptive divergence trigger (volume-aware)
- Provenance tagging; exports (JSON, CSV, accepted TXT)
- Debug scripts and config hardening

## Outstanding
- Per-source weighting (boost BTC-only, downweight multi-asset wraps)
- Confidence shaping by source diversity and accepted count
- JSON schema versioning; unit tests for drop_outliers, trimmed_mean, adaptive_trigger
- Optional: CoinTelegraph RSS integration
- Optional: Relaxed JSON extraction fallback for Perplexity when strict JSON not returned

------------------------

Tracer Bullet — Milestone Roadmap (with status)

Milestone V1 — Hardening & Reliability [DONE]
✓ Preflight/health checks (scripts/preflight.py; run.py --health)
✓ CLI wrapper with precedence (CLI > env > .env > defaults); early logging
✓ Robust Telegram delivery (plain text default, truncation, 200/400/429 handling, TB_NO_TELEGRAM)
✓ Artifact retention (prune runs/ and bars/ by mtime; TB_ARTIFACTS_KEEP)
✓ Centralized logging utilities; DEBUG/INFO modes
✓ Tests & CI (pytest suite; lint/format checks documented)
✓ Docs: README, RUNBOOK, CONTRIBUTING, dev_logs, .env.example, docs/payload

Milestone V2 — Crowd Immunity [DONE]
✓ Source Diversity Engine (diversity.py; confidence shaping, echo-penalties; payload.source_diversity)
✓ Cascade/HYPE Detector (cascade.py; repetition vs quant confirm; payload.cascade_detector)
✓ Contrarian Viewport (contrarian.py; informational tag; payload.contrarian_viewport)
✓ Telegram one-liners for diversity/cascade/contrarian when applicable
✓ Tests for diversity, cascade, contrarian; schema presence

Milestone V3 — Bias Immunity + Sizing [PLANNED]
- Multi-timescale scoring:
  - Compute short/mid/long horizon metrics (sentiment, divergence, volume_z)
  - Combine with transparent weights; require alignment for high confidence
  - payload.timescale_scores = {short, mid, long, combined}
- Negative-confirmation weighting:
  - Deduct confidence for specific contradictions (bounded, explainable)
  - payload.confirmation_checks = [{check, passed, delta}], with clamped total
- Confidence → position sizing:
  - Map confidence to target_R with floors/caps and optional vol-normalization
  - payload.position_sizing = {confidence, target_R, caps_applied, notes}
  - Telegram line only if above floor
- Tests: blending math, penalties clamping, sizing boundaries; schema tests

Milestone V3.1 — Multi-Asset Foundations (Crypto + Stocks) [DONE]
- Universe configuration (config/universe.yaml) for crypto and stocks
- Stock bars adapter (e.g., Polygon/AlphaVantage/Yahoo) with same interface
- Trading-hours awareness (RTH vs extended; volume_z handling)
- Multi-asset orchestrator (scripts/scan_universe.py), ranking Top N
- Universe digest to Telegram (compact per-asset summary)
- Tests: universe loader, symbol utils, bars_stock adapter (fixture), orchestrator ranking, schema extensions

Milestone V4 — Backtesting & Learning Loop [PLANNED]
- Backtest/replay harness using bars + runs payloads (event studies, KPIs)
- Outcome labeling in SQLite (forward return windows, hit-rate)
- First supervised model to calibrate confidence/edge (explainable baseline)
- Model monitoring and drift checks; ablation tests
- Tests for data prep, labeling, and calibration pipeline

Milestone V4.1 — Execution & Risk (Dry-Run) [PLANNED]
- Paper-trade execution simulator (order lifecycle, slippage model)
- Portfolio tracking (PnL, exposure, risk caps)
- Kill-switches/guardrails and dry-run audit logs
- Tests for sizing→orders→fills and portfolio accounting sanity

Milestone V5 — Autonomy & Scale [PLANNED]
- Live broker integration (e.g., Alpaca) behind feature flag
- Multi-asset risk budgeting and rebalancing
- Continuous/periodic retraining, feature store, drift-aware deployment
- Horizontal scale (priority queues, smart sampling, cost-aware scheduling)
- Observability: metrics dashboard for confidence/divergence/vol_z, model health

Cross-Cutting (Ongoing)
- Documentation: architecture overview, payload schema updates, runbook ops
- Observability: per-run metrics logs, compact CSV for quick plots
- Safety: strict caps, fallback paths, and graceful degradation

-----------------------------
# Tracer Bullet — Comprehensive Roadmap Update (what’s done, what’s planned, and alignment)

Below is a consolidated, milestone-based roadmap compiled from our conversation history, organized by version, with explicit completion status, what shipped in each version, what’s next, and a clear alignment verdict versus the initial vision.

## Executive verdict

- Alignment: The project remains aligned with the original “Tracer Bullet” approach and the objective to build an interpretable, evidence-driven alpha engine that blends narrative with price/volume and ships incrementally with auditability and guardrails. We are not building “something else”; we’ve deepened exactly what we set out to do: reliability first, explainability, multi-asset foundations, and human-readable outputs.  
- Scope adjustments: Two timeline corrections were made for clarity, not direction changes:  
  - 24/7 cloud scheduling is now explicitly a later milestone (v6) after full testing, rather than earlier.  
  - v3.3 expands to all available crypto alts (not a small subset).  

## Completed milestones

### v1 — Hardening & Reliability [DONE]
- Preflight/health checks to validate environment and Telegram reachability; automatic directory setup for artifacts.  
- CLI wrapper with clear precedence (CLI > env > .env > defaults), centralized structured logging, INFO/DEBUG modes.  
- Robust Telegram delivery: plain-text default, truncation safeguards, graceful handling of 200/400/429, opt-out via flag.  
- Artifact retention: pruning runs/ and bars/ by most-recent N files; configurable via env.  
- Tests and CI: unit tests for divergence, payload schema, Telegram formatting, directory checks; CI lint/test flow.  
- Documentation: README, RUNBOOK, CONTRIBUTING, .env.example, payload docs.  

Why this matters: Establishes a dependable, reproducible core loop with safe ops, visibility, and clean artifacts.

### v2 — Crowd Immunity [DONE]
- Source Diversity Engine: shapes confidence by unique sources and penalizes “echo chambers”; artifacts added to payload.  
- Cascade/HYPE Detector: flags repetitive narrative lacking quant confirmation; bounded confidence deltas; payload enriched.  
- Contrarian Viewport: informational tag for potential crowd mistakes under certain narrative/price conditions; included in payload and one-liners.  
- Tests for diversity/cascade/contrarian presence and behavior.

Why this matters: Reduces herd-driven noise; improves robustness and interpretability of narrative signals.

### v3 — Bias Immunity + Sizing [DONE]
- Multi-timescale scoring: short/mid/long metrics and combined view with alignment gating.  
- Negative-confirmation checks: structured penalty logic with clamps; transparent reasons in payload.  
- Informational position sizing: confidence-to-R mapping, optional vol-normalization; reported without forcing trades.  
- Telegram lines reflect timescales, penalties, and sizing guidance; tests cover blending, clamps, and boundaries.

Why this matters: Adds disciplined structure to confidence, avoids over-trust in contradictory evidence, and connects confidence to position logic.

### v3.1 — Multi-Asset Foundations (Crypto + Stocks) [DONE]
- Universe configuration for symbols; symbol utilities for normalization and type detection.  
- Trading-hours awareness (crypto 24/7 vs equities RTH/extended/closed).  
- Stock bars adapter scaffold (with safe fallbacks), orchestration for multi-symbol scan, top-N ranking, digest utilities.  
- Payload extensions: symbol_type, market_hours_state, timescale presence.  
- Universe runs written to universe_runs/ with timestamping; optional mirroring to runs/.  
- Git integration hooks implemented behind env gates (off by default) for mirror/commit/push; robust try/except and logging.  
- Tests: universe loader, symbol utils, trading hours, stock adapter shape/safety, ranking determinism.

Why this matters: Enables consistent multi-asset scanning and reporting without breaking single-asset flow.

### v3.1.x — Human Digest Integration (crypto-first, number-free) [DONE]
- Added number-free, conversational digest formatter producing a consistent crypto-first report (BTC/ETH prioritized), including levels-to-watch (descriptive), entries/exits, and risk-based sizing bands.  
- Integrated into single-run flow: produced after artifacts are written, sent to Telegram, and printed to console; analyzer logic remains unchanged.  
- Optional prompt/style reference file; optional toggle to enable/disable digest output; safe defaults preserved.  

Why this matters: Delivers a human-ready narrative output without exposing raw metrics, boosting usability for decision-making while keeping the quantitative engine intact.

## In progress

### v3.1.x — Auto-commit/push hardening [IN PROGRESS]
- Goal: Ensure universe_runs/*.json and universe_runs/metrics.csv are staged, committed, and pushed automatically when env gates are on.  
- Current status: Commit/push plumbing exists behind env flags, but defaults are OFF; some terminals may miss staging for metrics.csv; logs need explicit “Auto-commit done.” / “Pushed.” confirmations.  
- Next steps: Confirm staging includes both JSON and metrics; add explicit result logs; verify non-interactive push across environments.

Why this matters: Eliminates manual staging/pushing and keeps repo artifacts consistent across runs and environments.

## Planned milestones

### v3.2 — Reliability Hardening (agent, non-24/7)
- Retries/backoff for transient providers; structured error handling and graceful skips.  
- Schema checks and digest self-check for thin or missing inputs; produce useful outputs even when evidence is sparse.  
- Clear alert notes when runs are skipped or degraded.

Why this matters: Improves run resilience and developer/operator trust before moving to continuous scheduling.

### v3.3 — Full Crypto Alt Coverage + Evidence Lines
- Expand coverage to all available liquid crypto alts (not just a few), using the same number-free template.  
- Add brief “why now” evidence lines per BTC/ETH and key alts, describing sentiment/news/structure vs price in plain English (no numbers, no links).  
- Maintain crypto-first priority and keep equities de-emphasized.

Why this matters: Completes crypto breadth while preserving interpretability, providing rationale for attention and bias.

### v3.4 — Execution Quality (paper/dry-run)
- Microstructure-aware tactics (market vs limit vs slices by spread/volatility) and cool-downs to avoid clustering.  
- Volatility-aware sizing with conservative caps.  
- Measure slippage versus baseline to confirm improvements.

Why this matters: Turns good signals into better realized outcomes while staying in a safe, non-live mode.

### v4 — Backtesting & Governance
- Event-ordered replay for bars+headlines; walk-forward validation; cohort analytics (asset/time-of-day/volatility regime).  
- Parameter governance cadence with documented thresholds from out-of-sample.  
- Reproducible backtests with clear in/out-of-sample splits.

Why this matters: Converts plausible intuition into evidence-backed settings and reduces hidden look-ahead risk.

### v4.1 — Paper Execution & Risk Controls
- Paper order lifecycle with audit logs; portfolio caps; kill-switches and guardrails.  
- Idempotency, reproducibility tags per decision/version.

Why this matters: Operational discipline before any live risk, ensuring safe failure modes.

### v5 — Data Breadth & Explainability+
- Optional attention/crowd proxies as secondary evidence (controlled, never primary).  
- Source credibility learning; compact case files per signal for audits/postmortems.

Why this matters: Improves precision and review speed without sacrificing interpretability.

### v6 — 24/7 Cloud Agent Run (after full testing)
- GitHub Actions (or equivalent) scheduled workflows: crypto-only every 15 minutes; mixed hourly with staggered minute.  
- Secrets management; non-interactive push; deterministic cadence; Telegram delivery.  
- Monitoring/rollback for the scheduler jobs.

Why this matters: Moves to truly autonomous operation only after we’ve finished hardening, coverage, and testing.

### v7 — Live Capital (small, guarded)
- Strict loss limits, circuit breakers, anomaly alerts, version tagging; limited deployment scope.  
- Rollback rehearsed; postmortem-ready artifacts.

Why this matters: Begin live exposure safely, learning from real frictions without over-scaling.

## Are we on plan?

- Yes, with a clarified timeline: Up through v3.1.x we are on track and consistent with the tracer-bullet philosophy—thin end-to-end, then harden, then expand coverage, then automate scheduling, then consider live.  
- The only course correction was to explicitly place 24/7 scheduling at v6 after testing, and to broaden v3.3 to cover all available alts; both are alignment fixes, not directional changes.

## Operational notes (scheduling and automation guardrails)

- When we reach v6, scheduled workflows can use cron-based triggers with sensible intervals; GitHub Actions supports 5-minute minimum cadence and may delay around top-of-hour loads, so staggered minutes are recommended to reduce contention[1][2][3].  
- If we explore interim in-app scheduling for dev or server use, APScheduler’s cron/interval triggers and background schedulers are a robust option before moving to managed schedules[4][5][6][7][8].  

## What to do next (immediate focus)

- Finish v3.1.x hardening: confirm broadened non‑.py staging covers `runs/*.json`, `universe_runs/metrics.csv`, `eval_runs/*`; logs show commit/push summaries.  
- Start v3.2: implement retries/backoff and schema/digest self-checks; ensure graceful degradation and actionable logs when inputs are thin.  
- Prepare v3.3 backlog for “all alts + evidence lines” with the digest template unchanged in tone and structure.

These steps preserve our reliability-first approach and set us up for a smooth v6 shift to 24/7 automation after full testing.

Sources
[1] Why does my cron configured GitHub Action not run every 2 minutes? https://stackoverflow.com/questions/63192132/why-does-my-cron-configured-github-action-not-run-every-2-minutes
[2] How to Schedule Workflows in GitHub Actions - DEV Community https://dev.to/cicube/how-to-schedule-workflows-in-github-actions-1neb
[3] Run your GitHub Actions workflow on a schedule - Jason Etcovitch https://jasonet.co/posts/scheduled-actions/
[4] User guide — APScheduler 3.11.0.post1 documentation https://apscheduler.readthedocs.io/en/3.x/userguide.html
[5] Job Scheduling in Python with APScheduler | Better Stack Community https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/
[6] Scheduled Jobs with Custom Clock Processes in Python with ... https://devcenter.heroku.com/articles/clock-processes-python
[7] Python Job Scheduling: Methods and Overview in 2025 https://research.aimultiple.com/python-job-scheduling/
[8] I Replaced Cron Jobs with Python Schedulers | by Muhammad Umar https://python.plainenglish.io/i-replaced-cron-jobs-with-python-schedulers-6a25f94bd642
[9] Tracer Bullets - C2 wiki https://wiki.c2.com/?TracerBullets
[10] How Tracer Bullets Speed Up Software Development | Built In https://builtin.com/software-engineering-perspectives/what-are-tracer-bullets
[11] Tracer-Bullet — Why we should build features during discovery https://thedigitalbusinessanalyst.co.uk/tracer-bullet-why-we-must-build-features-during-discover-952df9c5a65b
[12] bullet-scraper/scrapes/afbulletsafe.txt at master - GitHub https://github.com/AF-VCD/bullet-scraper/blob/master/scrapes/afbulletsafe.txt
[13] Caitlin Hudon - Tracer bullets + working backwards - YouTube https://www.youtube.com/watch?v=vNZY0zhg3Do
[14] How do you make a workflow run randomly within a given time period? https://github.com/orgs/community/discussions/131450
[15] [PDF] Go: Building Web Applications - anarcho-copy https://edu.anarcho-copy.org/Programming%20Languages/Go/Go%20building%20web%20application.pdf
[16] [PDF] EXPRESSION OF INTEREST - BECIL https://www.becil.com/uploads/topics/17193916113963.pdf
[17] GitHub Actions Cron Schedule for Running Once in 2 Weeks #158356 https://github.com/orgs/community/discussions/158356
[18] The Evolving Landscape of Antibody–Drug Conjugates: In Depth ... https://pubs.acs.org/doi/10.1021/acs.bioconjchem.3c00374
[19] [PDF] DOT&E FY2021 Annual Report https://www.dote.osd.mil/Portals/97/pub/reports/FY2021/other/2021DOTEAnnualReport.pdf
[20] Apscheduler is skipping my task. How to eliminate this? https://stackoverflow.com/questions/73343854/apscheduler-is-skipping-my-task-how-to-eliminate-this

------------------------------
updated roadmap

Here’s the revised, full, end-to-end, milestone-based roadmap, incorporating everything we’ve shipped, what’s in progress, what’s planned, and exactly where the Polymarket read-only viewport fits. It respects your directive: 24/7 scheduling at v6 (after full testing), v3.3 covers all available crypto alts, and execution on Polymarket is out-of-scope until ≥6 months of evaluation.

Executive summary
- We are on plan with the tracer-bullet philosophy: thin end-to-end → harden → breadth → evidence → evaluate → automate → (later) live.
- Completed through v3.1.x (including human digest). Auto-commit/push hardening is in progress.
- Polymarket read-only viewport is added in v3.3; evaluation in v3.4; any scheduling only at v6; live bets only after ≥6 months if metrics justify.

Versioned milestones (with status)

v1 — Hardening & Reliability [DONE]
- What:
  - Preflight/health checks (creates runs/ and bars/, Telegram reachability).
  - CLI precedence (CLI > env > .env > defaults), centralized logging (INFO/DEBUG).
  - Robust Telegram delivery (plain text default, truncation, 200/400/429 handling), TB_NO_TELEGRAM.
  - Artifact retention: prune runs/ and bars/ by mtime via TB_ARTIFACTS_KEEP.
  - Tests & CI (divergence, payload schema, Telegram, dir checks; lint/format).
  - Docs: README, RUNBOOK, CONTRIBUTING, .env.example, payload docs.
- Why: Stable, auditable baseline.

v2 — Crowd Immunity [DONE]
- What:
  - Source Diversity Engine (confidence shaping, echo penalties).
  - Cascade/HYPE Detector (repetition vs quant confirmation; bounded deltas).
  - Contrarian Viewport tag (informational).
  - Telegram one-liners; tests for modules and schema presence.
- Why: Reduce herd noise; keep confidence explainable.

v3 — Bias Immunity + Sizing [DONE]
- What:
  - Multi-timescale scoring (short/mid/long + combined with alignment requirement).
  - Negative-confirmation checks with clamps; transparent reasons.
  - Informational position sizing (confidence→R, floors/caps; optional vol-normalization).
  - Telegram lines for timescales/penalties/sizing.
  - Tests for blending/clamps/boundaries.
- Why: Disciplined, interpretable confidence and sizing guidance.

v3.1 — Multi-Asset Foundations (Crypto + Stocks) [DONE]
- What:
  - Universe config (config/universe.yaml); symbol normalization/type detection.
  - Trading-hours awareness (crypto 24/7; equities RTH/EXT/CLOSED).
  - Stock bars adapter scaffold with safe fallbacks.
  - Orchestrator (scan_universe.py): fan-out, rank Top-N (deterministic), digest_utils.
  - Payload additions: symbol_type, market_hours_state.
  - Universe artifacts: universe_runs/*.json and metrics.csv; optional mirror to runs/.
  - Git hooks behind env gates (OFF by default): mirror/commit/push; robust try/except and logging.
  - Tests: universe loader, symbol utils, trading hours, adapter shape/safety, ranking determinism.
- Why: Scale to multi-asset without breaking single-asset flow.

v3.1.x — Human Digest Integration (crypto-first, number-free) [DONE]
- What:
  - Digest formatter (scripts/digest_formatter.py) producing crypto-first, conversational, number-free digest:
    - BTC/ETH prioritized; entries/exits; levels-to-watch (descriptive); risk bands; Playbook footer; equities de-emphasized.
    - Qualitative bands from internal signals (polarity, confidence band, volume state, alignment, quality tags, readiness, sizing).
  - Integrated post-artifact write; Telegram + console; optional toggle/env.
- Why: Immediate human usability without exposing raw metrics.

v3.1.x — Auto-commit/push hardening [IN PROGRESS]
- What (to finish):
  - Ensure staging includes universe_runs/*.json and universe_runs/metrics.csv.
  - Explicit logs: “[Universe] Auto-commit done.” / “[Universe] Pushed.” or clear skip reason.
  - Verify non-interactive push across environments (SSH/PAT/credential helper).
- Why: Fully hands-off artifact versioning when enabled; defaults remain safe (OFF).

v3.2 — Reliability Hardening (agent, non-24/7) [PLANNED]
- What:
  - Retries/backoff, timeouts, and structured error categories for providers.
  - Schema and digest self-checks; graceful degradation for thin inputs.
  - Clear alert/skip notes; no crashes on partial data.
- DoD:
  - Transient failures recover; digest emits helpful fallbacks; logs actionable.
- Why: Strength before expanding breadth/outputs.

v3.3 — Full Crypto Alt Coverage + Evidence Lines + Polymarket (read-only) [PLANNED]
- Crypto breadth:
  - Include all available liquid crypto alts in the digest using the same number-free template.
  - One-liners per alt: bias/readiness/action; qualitative levels; brief rationale.
- Evidence lines:
  - BTC/ETH and key alts: 1–2 sentence “why now” in plain English (sentiment/news/structure vs price; no numbers).
- Polymarket (read-only) viewport:
  - Adapter (polymarket_adapter.py): list crypto-related markets (question, outcomes, implied probs, end_time, status, resolution source).
  - Bridge (polymarket_bridge.py): map TB’s qualitative read to stance (Engage/Stalk/Stand Aside), readiness (Now/Near/Later), edge label (market cheap/rich/in-line) expressed number-free; rationale line.
  - Digest: “Polymarket (crypto)” section with up to N curated markets (number-free).
  - Artifacts: store implied probs and TB internal probabilities for evaluation only; never shown in chat digest.
  - Toggles: TB_ENABLE_POLYMARKET=1, TB_POLYMARKET_MAX_ITEMS (default 3), TB_POLYMARKET_MIN_QUALITY.
- DoD:
  - Digest shows all alts; Polymarket section appears when enabled/quality met; artifacts carry eval fields.
- Why: Comprehensive crypto coverage + external odds comparison without execution or numeric leak.

v3.4 — Evaluation Pipeline (Polymarket + system reliability) [PLANNED]
- What:
  - Weekly evaluator: Brier score, log-loss, calibration curves, win-rate by edge label/cohorts; lead/lag analysis.
  - Event-ordered records; capture resolved outcomes; CSV/JSON summaries per week.
  - Reliability polish: ensure evaluation never breaks production runs; separate pipeline or safe post-process.
- DoD:
  - Stable weekly metrics; clear dashboards/tables; repeatable replay.
- Why: Build ≥6 months of evidence before any talk of live bets or scheduling.

v3.5 — Equities Bars Provider (background) [PLANNED]
- What:
  - Integrate a stable equities bar provider (Polygon/Alpaca/Yahoo) with throttling and fallbacks.
  - Remove fallback noise; improve equities structure read while keeping equities de-emphasized in digest.
- DoD:
  - Cleaner equities sections; no provider warning noise.
- Why: Quality uplift without changing crypto-first focus.

v4 — Backtesting & Governance (leak-free) [PLANNED]
- What:
  - Event-ordered replay of bars+headlines; lagged features only; forward returns evaluation windows.
  - Walk-forward validation; cohort analytics (asset, time-of-day, volatility regime, event type).
  - Parameter governance cadence; documented thresholds from OOS.
- DoD:
  - Reproducible backtests with clear splits; published settings with evidence.
- Why: Evidence-backed parameters; protect against look-ahead.

v4.1 — Paper Execution & Risk Controls (dry-run) [PLANNED]
- What:
  - Paper order lifecycle; portfolio caps; kill-switches/guardrails; audit logs; idempotency and version tags.
- DoD:
  - Operational discipline rehearsed; zero live risk; reproducible traces.
- Why: Prove operational readiness before live exposure.

v5 — Data Breadth & Explainability+ [PLANNED]
- What:
  - Optional attention/crowd proxies (secondary only).
  - Source credibility learning; compact “case files” per signal for audits/postmortems.
- DoD:
  - Higher precision without opacity; faster reviews.
- Why: Richer, explainable context while staying interpretable.

v6 — 24/7 Cloud Agent Run (after full testing) [PLANNED]
- What:
  - Scheduled workflows (GitHub Actions or equivalent):
    - Crypto-only every 15 minutes; mixed hourly (stagger minute).
  - Secrets management; contents:write permissions; non-interactive push.
  - Include read-only Polymarket viewport in schedule if metrics are satisfactory.
  - Monitoring, alerting, and rollback for scheduler jobs.
- DoD:
  - Cadence runs deterministic; artifacts/digests on schedule; safe rollback.
- Why: Autonomous operation only after reliability and evaluation are proven.

v7 — Live Capital (small, guarded) [PLANNED]
- What:
  - Strict daily loss limits, per-trade caps, circuit breakers, anomaly alerts, heartbeats; full version tagging.
  - If Polymarket ever advances to execution: separate, hard-gated adapter after ≥6 months of strong evaluation metrics; tiny notional pilot; separate risk caps.
- DoD:
  - Stable live trial; reproducible decisions; rollback rehearsed.
- Why: Learn real-world frictions safely.

Cross-cutting workstreams (ongoing)
- Documentation: keep README, RUNBOOK, ROADMAP, payload schema, digest spec, universe orchestrator docs, configuration, reliability, testing, git automation pages current.
- Observability: per-run metrics logs and compact CSV (universe_runs/metrics.csv), plus weekly evaluation outputs (v3.4).
- Safety: strict schema checks, graceful degradation, circuit-breakers for providers, TB_NO_TELEGRAM safe mode.
- Governance: version tags in artifacts and (later) in signals/trades; monthly review of parameters; quarterly regime review.

Immediate next actions
- Finish v3.1.x: staging for universe_runs/*.json + metrics.csv; explicit commit/push logs; validate non-interactive push.
- Start v3.2: retries/backoff, timeouts, schema/digest self-checks, graceful skips, alert notes.
- Prep v3.3 backlog:
  - All alts coverage in digest (template blocks and symbol list expansion).
  - Evidence lines (plain-English “why now”).
  - Polymarket read-only: adapter, bridge, digest section, artifact fields, toggles.

Definition of done roll-up
- v3.1.x: Human digest integrated [DONE]; auto-commit/push hardened [IN PROGRESS].
- v3.2: Runs resilient under transient failures; digest self-checks; clear skips.
- v3.3: All liquid alts appear; evidence lines added; Polymarket read-only section live; artifacts include evaluation fields (no numbers in chat).
- v3.4: Weekly Brier/log-loss/calibration delivered; event-ordered; robust.
- v6: 24/7 schedule active only after above proofs; rollbacks tested.

This is the authoritative plan going forward. If you want, I can also generate a one-page “Now → Near → Next” visual and add precise acceptance checklists per milestone.

--------------------------------------------------------------

Updated Roadmap 1.0

Here’s the fully revised, very detailed, crypto-only, milestone-based roadmap with the renumbering you approved:
- v4 = Data Breadth & Explainability+ (moved up from old v5)
- v5 = 24/7 Cloud Agent Run (from old v6)
- v6 = Paper Execution & Risk Controls (from old v4.1; starts after v5 cadence is proven)
- v7 = Live Capital
Polymarket is read-only and limited to BTC/ETH in v3.3; evaluation in v3.4. Stocks are removed/parked.

Status summary
- Done: v1, v2, v3, v3.1, v3.1.x (digest shipped; reliability hardening in progress)
- Current focus: finish v3.1.x (auto-commit/push + retries/backoff/self-checks)
- Next: v3.3 (all-liquid crypto alts + evidence lines + Polymarket BTC/ETH read-only), v3.4 (evaluation), then v4 (explainability+), v5 (24/7), v6 (paper), v7 (live)

Milestones in detail

v1 — Hardening & Reliability [DONE]
- Scope:
  - Preflight/health checks; creates runs/ and bars/; validates Telegram; fail-fast on misconfig
  - CLI precedence (CLI > env > .env > defaults); centralized logging; INFO/DEBUG
  - Telegram robustness: plain text default, 4k truncation, 200/400/429 handling; TB_NO_TELEGRAM
  - Artifact retention: prune runs/ and bars/ by mtime; TB_ARTIFACTS_KEEP
  - Tests & CI: divergence, payload schema, Telegram formatting, dir checks; lint/format
  - Docs: README, RUNBOOK, CONTRIBUTING, .env.example, payload docs
- DoD: Fresh checkout runs clean; CI green
- Instrumentation: structured logs; retention counters; Telegram send status

v2 — Crowd Immunity [DONE]
- Scope:
  - Source Diversity Engine (unique sources; echo penalties; confidence cap)
  - Cascade/HYPE Detector (repetition vs quant confirm; bounded deltas; hype tag)
  - Contrarian Viewport (informational tag)
  - Telegram one-liners; tests and schema presence
- DoD: Fields present; tests green

v3 — Bias Immunity + Sizing [DONE]
- Scope:
  - Multi-timescale scoring (short/mid/long + combined with alignment gate)
  - Negative-confirmation checks with clamp; transparent reasons
  - Informational sizing (confidence→R; floors/caps; optional vol-normalization)
  - Telegram lines for timescales/penalties/sizing
  - Tests for blending/clamps/boundaries
- DoD: Payload fields populated; digest lines when applicable; tests green

v3.1 — Crypto Foundations (universe) [DONE]
- Scope:
  - Crypto-only universe config (config/universe.yaml)
  - Symbol utils for crypto normalization/type detection
  - Orchestrator (scan_universe.py): fan-out; deterministic top-N ranking
  - Artifacts: universe_runs/*.json + universe_runs/metrics.csv; optional mirror to runs/
  - Git hooks (OFF by default): TB_UNIVERSE_MIRROR_TO_RUNS, TB_UNIVERSE_GIT_AUTOCOMMIT, TB_UNIVERSE_GIT_PUSH (robust try/except, clear logs)
  - Tests: universe loader; symbol utils; ranking determinism; adapter safety
- DoD: Multi-symbol crypto run emits artifacts; ranking stable; tests green

v3.1.x — Human Digest + Reliability Hardening [IN PROGRESS]
- Shipped:
  - Human digest: crypto-first (BTC/ETH prioritized), number-free; entries/exits; levels-to-watch (descriptive); risk bands; Playbook; Telegram + console; TB_HUMAN_DIGEST toggle
- To finish:
  - Auto-commit/push hardening:
    - Stage explicitly: universe_runs/*.json and universe_runs/metrics.csv
    - Clear logs: “[Universe] Auto-commit done.” / “[Universe] Pushed.”; explicit skip reasons (env off, nothing to commit, auth failure)
    - Verify non-interactive push (SSH or PAT/credential helper)
  - Core reliability:
    - Retries/backoff + timeouts for external fetches
    - Schema self-checks; digest self-checks for thin inputs; graceful skips with alert notes
- DoD:
  - 3-day burn-in: zero crashes; <1% runs degraded with helpful digest notes
  - If enabled, commit/push logs consistent; repo reflects cadence

v3.3 — Full Crypto Alts (phased) + Evidence Lines + Polymarket BTC/ETH (read-only) [PLANNED]
- Crypto breadth (phased):
  - Include all liquid alts in artifacts; digest surfaces top-K by rank/quality (config-gated rollout to avoid noise)
- Evidence lines:
  - BTC/ETH + top alts get a 1–2 sentence “why now” (sentiment/news/structure vs price), strictly number-free
- Polymarket (read-only, BTC/ETH only):
  - Adapter: discover BTC/ETH markets with strict filters (clear resolution source, near/mid-dated, minimum-liquidity proxy)
  - Bridge: map TB qualitative read to stance (Engage/Stalk/Stand Aside), readiness (Now/Near/Later), edge label (market cheap/rich/in-line); rationale (no numbers)
  - Digest: “Polymarket (crypto)” section with up to 2 markets (BTC/ETH)
  - Artifacts: store implied probs and TB-internal probabilities for evaluation only
  - Toggles: TB_ENABLE_POLYMARKET=1; TB_POLYMARKET_MAX_ITEMS=2; TB_POLYMARKET_MIN_QUALITY
- DoD: Digest shows top-K alts with evidence lines; Polymarket BTC/ETH appears when filters pass; artifacts carry eval fields

v3.4 — Evaluation Pipeline (Polymarket + System) [PLANNED]
- Scope:
  - Weekly evaluator (manual or non-24/7 schedule):
    - Brier score, log-loss, calibration curves for Polymarket BTC/ETH predictions (TB internal p vs resolved outcomes)
    - Win-rate by edge label/cohorts; lead/lag vs odds convergence
  - Event-ordered snapshots; resolved outcomes captured; reproducible weekly CSV/JSON
  - Isolation so evaluation cannot break production runs
- DoD: Weekly metrics emitted; ≥50 resolved observations before drawing conclusions

v4 — Data Breadth & Explainability+ [PLANNED]  (renumbered from old v5)
- Scope:
  - Optional attention/crowd proxies (secondary only; never primary gates)
  - Source credibility learning (lightweight, interpretable reweighting over time)
  - Compact per-signal “case files” (human-readable bundles for audits/postmortems)
- DoD: Precision uplift on targeted cohorts; faster audits; preserve interpretability

v5 — 24/7 Cloud Agent Run (after full testing) [PLANNED]  (renumbered from old v6)
- Scope:
  - Scheduled workflows (GitHub Actions or equivalent):
    - Crypto-only cadence every 15 minutes; stagger minute to avoid top-of-hour congestion
  - Secrets: Telegram token/chat; push rights (GITHUB_TOKEN contents:write or PAT/SSH)
  - Include Polymarket BTC/ETH viewport if v3.4 metrics are stable; still read-only
  - Monitoring + rollback: failure alerts, retries, manual disable switch, safe stop
- DoD: Deterministic cadence for multi-weeks; artifacts and digests on schedule; rollback rehearsed

v6 — Paper Execution & Risk Controls (dry-run) [PLANNED]  (renumbered from old v4.1; starts after v5)
- Scope:
  - Paper order lifecycle simulator (market/limit/slice by spread/volatility); idempotent order IDs; audit logs
  - Portfolio caps; per-asset exposure limits; circuit breakers; kill-switches
  - Version tags (code/prompt/params) embedded in every decision trace
  - Runs on the same 24/7 cadence established in v5
- DoD: Stable dry-run at 24/7 cadence; consistent audit logs; caps/guards verified

v7 — Live Capital (small, guarded) [PLANNED]
- Scope:
  - Strict daily loss limits; per-trade caps; circuit breakers; anomaly alerts; heartbeats; full version tagging
  - If Polymarket execution is ever considered: only after ≥6 months strong evaluation; hard-flagged; tiny notional; separate caps; dedicated monitoring
- DoD: 4–6 week stable live trial; reproducible decisions; rollback rehearsed

Cross-cutting (ongoing)
- Docs: keep Roadmap, Runbook, Payload Schema, Digest Spec, Universe Orchestrator, Configuration, Reliability, Testing, Git Automation current
- Observability: per-run metrics CSV (universe_runs/metrics.csv); weekly evaluation outputs (v3.4); degraded-run markers
- Safety: strict schema/versioning; graceful degradation; TB_NO_TELEGRAM safe mode; provider circuit breakers
- Governance: version tags in artifacts; monthly parameter review; quarterly regime review; change logs

Immediate next actions
- Finish v3.1.x:
  - Stage universe_runs/*.json + universe_runs/metrics.csv; add explicit commit/push logs; verify SSH/PAT push
  - Add retries/backoff + timeouts; schema/digest self-checks; graceful skip alerts
  - 3-day burn-in; target <1% degraded runs
- Prep v3.3:
  - Config-gate all-liquid alts; define top-K digest surface
  - Implement evidence line generator template (plain-English)
  - Polymarket BTC/ETH-only: adapter, bridge, strict filters; artifact-only numeric fields; digest section; TB_ENABLE_POLYMARKET flag
- Scaffold v3.4:
  - Define evaluator schema, cohorts, and weekly CSV/JSON outputs; event-ordered archive layout

This is the authoritative, renumbered roadmap reflecting all confirmed updates.

--------------------------------------------

Alright — here’s the **full authoritative, renumbered roadmap** with the two new milestones (**v4.2 Backtesting & Governance** and **v4.3 Reliability Ops Hardening**) explicitly included, and all current status noted.  
This supersedes the previous version — you can treat this as the single source of truth going forward.  

***

# **Tracer Bullet — Authoritative Milestone‑Based Roadmap (Crypto‑Only)**  
*(Polymarket read‑only BTC/ETH in v3.3, live execution deferred; stocks removed/parked)*

***

## **Status Summary**
- **✅ Done**: v1, v2, v3, v3.1, v3.1.x (human digest shipped; some reliability items moved to v4.3)  
- **🚧 Current focus**: Finish v3.1.x ops hardening → stage/push polish + retries/backoff/self‑checks (now v4.3)  
- **📅 Next major features**: v3.3 (all‑liquid alts + evidence lines + Polymarket BTC/ETH read‑only), v3.4 (evaluation), then v4 series.

***

## **Detailed Milestones**

### **v1 — Hardening & Reliability** ✅  
**Scope**
- Preflight/health checks; directory creation; Telegram reachability.  
- CLI precedence (CLI > env > .env > defaults), centralized structured logging.  
- Robust Telegram delivery (plain text default, truncation safeguard, 200/400/429 handling).  
- Artifact retention (`runs/`, `bars/`) by mtime via `TB_ARTIFACTS_KEEP`.  
- Tests & CI for divergence calc, schema validation, Telegram, directory checks, lint/format.  
- Documentation: README, RUNBOOK, CONTRIBUTING, payload spec.  

**DoD**: Fresh checkout runs clean; CI green; artifacts + logs stable.

***

### **v2 — Crowd Immunity** ✅  
**Scope**
- **Source Diversity Engine**: unique source count, echo penalties, confidence cap.  
- **Cascade/HYPE Detector**: repetition vs quant confirmation; bounded deltas.  
- **Contrarian Viewport**: tag potential crowd mistakes.  
- Telegram one‑liners for above; schema/tests in place.

**DoD**: Fields populated in payload; tests pass.

***

### **v3 — Bias Immunity + Sizing** ✅  
**Scope**
- Multi‑timescale scoring (short/mid/long) + combined w/ alignment gating.  
- Negative‑confirmation penalties (clamped), with transparent reasons in payload.  
- Confidence → R mapping, floors/caps, optional vol‑normalization.  
- Telegram lines for timescales/penalties/sizing; blending/clamp/boundary tests.

**DoD**: Informational sizing stable; tests pass.

***

### **v3.1 — Crypto Foundations (Universe)** ✅  
**Scope**
- Crypto‑only universe config (`config/universe.yaml`); symbol utils for type detect/normalize.  
- Orchestrator: multi‑symbol fan‑out, deterministic Top‑N ranking.  
- Artifacts: `universe_runs/*.json`, `metrics.csv`; optional mirror to `runs/`.  
- Git hooks (OFF by default): mirror/commit/push with safe logging.  
- Tests for loader, utils, ranking stability, adapter safety.

**DoD**: Multi‑symbol run stable; artifacts complete; tests pass.
Got it ✅ — we can fold the **“Telegram alternative” work** directly into the **v3.1.x** milestone so it’s part of the current line‑of‑effort, instead of pushing it to a later version.

Here’s how the updated **v3.1.x** scope will look:

***

 **v3.1.x — Human Digest + Delivery Hardening** *(updated)*

**Scope**
- **Crypto‑only Human Digest** *(already shipped)*:
  - BTC/ETH prioritised; full TFs (1h, 4h, 1D, 1W, 1M).
  - Stocks hidden from TG but kept in artifacts.

- **Telegram enhancements** *(already live)*:
  - Multi‑part send with [i/N] headers.
  - Weekly/Engine sections preserved.

- **🚀 New: Alternative delivery channel**:
  - Integrate **Discord webhook sender** with rich embeds for full digest delivery without size truncation.
  - Map digest sections to Discord embeds:
    - Embed 1: Header + Executive Take
    - Embed 2: Weekly + Engine
    - Embeds 3+: One per asset with all TFs
  - Automatic splitting across multiple messages if embeds exceed limits (≤10 embeds / ≤6,000 chars per message).
  - Configurable via `TB_ENABLE_DISCORD` and `DISCORD_WEBHOOK_URL` env vars.
  - Keep TG send in parallel (for short-form digest or redundancy) while Discord becomes primary full‑length channel.

- **Ops hardening** *(already planned in 3.1.x)*:
  - Auto‑commit/push artifacts (`universe_runs/*.json` + `metrics.csv`); explicit logs for commit/push.
  - Reliability layer with retries/backoff and schema/digest self‑checks.
  - Graceful degraded‑run notes.

**DoD**
- Discord channel receives full, single‑send digest (no truncation, all TFs).
- TG still receives shortened version (or crypto‑only as per flags) if enabled.
- 3‑day burn‑in: zero crashes, <1% degraded runs, consistent artifact pushes.



***

### **v3.1.x — Human Digest (Crypto‑Only)** ✅ (core shipped)  
**Scope**
- TG digest: **crypto‑only** (BTC/ETH prioritised), full TFs (1h, 4h, 1D, 1W, 1M).  
- Stocks hidden from TG but present in artifacts.  
- Multi‑part TG sending with [i/N] headers.  
- Weekly/Engine sections preserved; provider rotation unchanged.

**Shipped**: Above features live.  
**Reliability tasks → moved to v4.3**.

***

### **v3.3 — Full Crypto Alts + Evidence Lines + Polymarket BTC/ETH Read‑Only** 📅  
**Scope**
- All liquid alts in artifacts; digest shows top‑K (config‑gated).  
- Evidence lines: BTC/ETH + top alts get 1–2 sentence “why now” (sentiment/news/structure; number‑free).  
- Polymarket adapter/bridge (BTC/ETH only): strict filters, stance/readiness/edge label, rationale; number‑free in TG, numeric in artifacts.  
- Toggles: `TB_ENABLE_POLYMARKET`, `TB_POLYMARKET_MAX_ITEMS`, `TB_POLYMARKET_MIN_QUALITY`.

**DoD**: Digest shows alts + evidence; Polymarket section appears when quality met.

***

### **v3.4 — Evaluation Pipeline (Polymarket + System)** 📅  
**Scope**
- Weekly evaluator: Brier score, log‑loss, calibration curves (BTC/ETH).  
- Cohort win‑rates; lead/lag vs odds convergence.  
- Event‑ordered snapshots and resolved outcome capture to CSV/JSON archives.  
- Evaluation isolated from production run stability.

**DoD**: ≥50 resolved obs before drawing conclusions; metrics reproducible.

***

### **v4 — Data Breadth & Explainability+** 📅  
**Scope**
- Optional attention/crowd proxies (secondary only).  
- Source credibility learning over time (lightweight, interpretable).  
- Compact per‑signal “case files” for audits/postmortems.

**DoD**: Detectable precision uplift; faster reviews; interpretability preserved.

***

### **v4.2 — Backtesting & Governance** 📅 *(newly explicit)*  
**Scope**
- Event‑ordered replay of bars + headlines; point‑in‑time features only.  
- Walk‑forward validation; rolling IS/OOS splits.  
- Cohort analytics by asset/time/volatility/event type.  
- Governance cadence: monthly parameter review/update.

**DoD**: No look‑ahead leaks; thresholds documented from OOS; cohort report produced.

***

### **v4.3 — Reliability Ops Hardening** 📅 *(newly explicit)*  
**Scope**
- **Auto‑commit/push polish**: Stage JSON + CSV, explicit commit/push logs, skip reasons, verify non‑interactive push.  
- **Core reliability**: Retries/backoff, timeouts, schema/digest self‑checks, graceful skips with alerts.  
- **Acceptance**: 3‑day burn‑in; <1 % degraded runs; zero crashes.

***

### **v5 — 24/7 Cloud Agent Run** 📅  
**Scope**
- Scheduled GH Actions (crypto‑only 15m cadence; staggered).  
- Secrets mgmt; deterministic cadence; monitored with rollback.  
- Include Polymarket BTC/ETH if v3.4 metrics are stable.

**DoD**: Multi‑week stable schedule; safe pause/rollback.

***

### **v6 — Paper Execution & Risk Controls** 📅  
**Scope**
- Dry‑run execution sim; portfolio caps; per‑asset limits; circuit breakers/kill‑switches.  
- Version tagging per decision; runs on v5’s cadence.

**DoD**: Stable dry‑run; guards verified.

***

### **v7 — Live Capital (Small, Guarded)** 📅  
**Scope**
- Small trial; strict loss limits; anomaly alerts; rollback rehearsed.  
- Polymarket execution only if ≥6mo strong eval; hard‑flags; tiny notional.

***

## **Cross‑Cutting Workstreams (Ongoing)**
- **Docs**: Keep roadmap/runbook/schema digest spec current.  
- **Observability**: Metrics CSV per run; weekly eval outputs; degraded‑run markers.  
- **Safety**: Schema versioning, graceful degradation, provider circuit breakers.  
- **Governance**: Version tags in artifacts; monthly param review; quarterly regime review.

***

## **Immediate Next Actions**
1. Finish **v4.3** tasks (auto‑commit/push + retries/backoff/self‑checks).  
2. Prep **v3.3** backlog (alts expansion, evidence lines, Polymarket adapter).  
3. Complete **v3.4** DoD (≥50 resolved obs, weekly snapshots, optional plots; evaluator MVP shipped).  

***

## **Now → Near → Next**

- **Now (active)**
  - v4.3 Reliability Ops Hardening (IN PROGRESS)
    - Retries/backoff, self‑checks, degraded‑run markers
  - v3.4 Evaluation Pipeline (IN PROGRESS)
    - DoD: accumulate ≥50 resolved observations; weekly snapshots; optional plots

- **Near (queued)**
  - v5 — 24/7 Cloud Agent Run (after reliability/evaluation proven)
    - Scheduled workflows, secrets, monitoring/rollback

- **Next (after Near)**
  - v6 — Paper Execution & Risk Controls (dry‑run)
    - Order lifecycle simulator, caps/limits, circuit breakers, kill‑switches

***

If you want, I can also give you a **Now → Near → Next** visual of this roadmap so it’s easier to track at a glance while we work through v4.3 → v3.3 → v3.4 in parallel. Want me to do that?



