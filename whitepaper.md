Tracer Bullet: A Market Intelligence Engine for Story–Price Divergence
Abstract
Tracer Bullet is a disciplined market intelligence system that detects and scores divergences between narrative (news/sentiment) and market behavior (price/volume). It emphasizes explainability, robustness, and operational reliability. The system introduces “Crowd Immunity” (diversity, cascade/hype detection, contrarian viewport), multi-timescale scoring, negative-confirmation weighting, and confidence-to-sizing mapping. Roadmap milestones extend to multi-asset scanning (crypto + equities), outcome-driven learning, and risk-aware execution.

1. Introduction
Motivation: Markets move on both facts (price/volume) and narratives (news, social, institutional commentary). Most systems overfit one dimension and are vulnerable to hype and noise.

Objective: Systematically detect story–price contradictions (divergences), quantify confidence, and provide transparent, risk-aware guidance.

Design principles:

Discipline over noise: default HOLD until thresholds are met.

Explainability: every adjustment is logged and auditable.

Robust operations: resilient messaging, artifact retention, tests/CI.

2. System Overview
Inputs:

Market: close/volume bars.

Narrative: curated headlines; relevance-filtered per symbol.

Core modules:

Sentiment and divergence scoring

Crowd Immunity (diversity/echo, cascade/hype, contrarian flag)

Timescale scoring (short/mid/long) and alignment

Negative-confirmation weighting (bounded, explainable penalties)

Confidence-to-sizing mapping (informational in V3)

Outputs:

Structured JSON payload with all reasoning

Concise Telegram summary

SQLite logging (for later backtesting and model training)

3. Data Pipeline
Ingestion:

Bars: rolling close/volume windows.

Headlines: source-annotated, relevance-scored; accept top-N, drop noise.

Normalization:

Symbol conventions; timestamp alignment; safe defaults for missing data.

Artifact retention:

Prunes runs/ and bars/ by mtime; configurable keep count.

4. Scoring Methodology
4.1 Narrative and Price Scores
Narrative:

Relevance-filtered headlines; sentiment aggregation; decayed narrative proxy.

Price:

Price move percent and volume_z; price_score via tanh mapping of signed move.

Divergence:

narrative_score − price_score; action thresholds guard entries.

4.2 Crowd Immunity
Source Diversity Engine:

Boost for multiple independent sources; echo penalties for concentration; net clamp.

Cascade/HYPE Detector:

Penalizes repetitive narratives lacking price/volume confirmation; tags “HYPE_ONLY” when triggered.

Contrarian Viewport:

Flags rare setups where extreme narratives meet flat/contradictory price.

4.3 Multi-Timescale Scoring (V3)
Horizons: short(60m), mid(180m), long(360m) using tail bars.

Per-horizon metrics: price_move_pct, volume_z, divergence_h.

Combined divergence: weighted sum with renormalized weights.

Alignment: 2/3 horizons agreeing beyond a magnitude threshold.

4.4 Negative-Confirmation Weighting (V3)
Checks (bounded penalties):

price_vs_narrative: notable short-term price move against combined divergence.

volume_support: low average volume_z for sizable divergence.

timescale_alignment: lack of alignment for sizable divergence.

Penalty clamp: total ≥ min_penalty and ≤ 0; applied to confidence.

4.5 Confidence → Position Sizing (Informational in V3)
Mapping:

Floor/cap band for confidence; linear interpolation to [min_R, max_R].

Optional volatility normalization (hook present; off by default).

Output is advisory only; no order placement in V3.

5. Decision Policy
Default state: HOLD unless both divergence magnitude and confidence exceed thresholds with adequate confirmations.

Escalation:

Alerts when divergence nears trigger or participation (volume_z) rises.

Tighten source filters if mixed-asset noise increases.

Invalidation rules:

Divergence sign flips over consecutive runs, or sentiment reversals exceed set bounds.

6. Reliability and Operations
CLI and health checks; early logging; environment precedence with .env support.

Telegram delivery hardening (plain text default, truncation, error handling, opt-out).

Tests and CI: unit tests for each module; schema-preserving payload checks.

Artifact retention and auditability: JSON payloads + DB entries per run.

7. Examples (Summarized)
Disciplined inaction:

Negative narrative vs neutral price; divergence below trigger; HOLD.

Confidence modest; penalties applied for weak volume support; explained in payload.

Timescale alignment:

3/3 horizons aligned; combined_divergence supportive; still below action threshold; sizing suggestion informational.

8. Roadmap
V3.1 Multi-Asset Foundations:

Universe config (crypto + stocks), stock bars adapter, market-hours awareness, orchestrator for Top-N ranking and Telegram digest.

V4 Backtesting and Learning:

Outcome labeling, replay harness, calibration from event studies, supervised baseline to refine confidence.

V4.1 Execution (Paper):

Order lifecycle simulation, slippage, PnL, portfolio caps, kill-switches.

V5 Autonomy and Scale:

Live brokerage integration behind flags, multi-asset risk budgeting, continual retraining, observability dashboards.

9. Safety, Risk, and Governance
Guardrails:

Confidence floors/caps; bounded penalties; explicit kill-switch paths.

Change management:

Feature flags for new behaviors; versioned payload schema.

Model risk (future phases):

Drift detection, ablation testing, rollbacks.

10. Conclusion
Tracer Bullet operationalizes a rational approach to market misperceptions by quantifying story–price divergence, filtering hype, and explaining every step. The current system is production-hardened and transparent; the roadmap scales it to a multi-asset, learning-driven alpha factory with risk-aware execution.

Appendix A: Key Environment Variables
Timescales: TB_TS_W_SHORT, TB_TS_W_MID, TB_TS_W_LONG

Confirmations: TB_CONF_PRICE_VS_NARR, TB_CONF_VOLUME_SUPPORT, TB_CONF_TS_ALIGN, TB_CONF_PENALTY_MIN

Sizing: TB_SIZE_CONF_FLOOR, TB_SIZE_CONF_CAP, TB_SIZE_MIN_R, TB_SIZE_MAX_R

Ops: TB_NO_TELEGRAM, TB_ARTIFACTS_KEEP, LOG_LEVEL

Appendix B: Payload Keys (Core + V3)
Core: accepted/rejected headlines, finbert_score, decayed_narrative, price_score, divergence, action, reason, summary/detail.

Crowd Immunity: source_diversity {unique, top_source_share, counts, adjustment}, cascade_detector {...}, contrarian_viewport.

V3: timescale_scores {short/mid/long, combined_divergence, aligned_horizons, alignment_flag, weights}, confirmation_checks, confirmation_penalty, position_sizing.