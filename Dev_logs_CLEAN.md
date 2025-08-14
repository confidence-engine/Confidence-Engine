# Tracer Bullet — Dev Logs (Clean)

A concise development history highlighting milestones, major features, testing, and operational controls.

---

## Milestones (high level)
- v1–v2: Thin-thread agent; end-to-end runs; Telegram DM prototype.
- v3.1: Multi-asset universe, crypto TF expansion, digest formatting/sending.
- v4.x (bridge): Polymarket providers/bridge via Perplexity Pro API, internal probability.
- v3.4: Evaluation pipeline — metrics, runner, ingest, weekly wrapper, auto-commit/push.
- v4.3 (in progress): Reliability hardening — retries/backoff, degraded-run markers, ops polish.

---

## Major features by area
- Universe scanning and artifacts
  - Multi-asset scan with crypto TFs (1h/4h/1D/1W/1M).
  - Enriched artifacts: evidence_line per asset; top-level `polymarket` array.
  - Auto-commit/push of universe artifacts (`TB_UNIVERSE_GIT_AUTOCOMMIT`, `TB_UNIVERSE_GIT_PUSH`).
- Digest delivery (Telegram/Discord)
  - Telegram splitter, crypto-only mode; Discord chunked embeds; strict no-send flags.
  - Number-free chat evidence; artifacts retain all metrics.
- Polymarket (PPLX-only)
  - `providers/polymarket_pplx.py`: key rotation, strict JSON parse, enforced `sonar` model.
  - `scripts/polymarket_bridge.py`: filtering/window/liquidity keywords, cap, internal_prob calibration.
  - Numbers gated in chat (`TB_POLYMARKET_NUMBERS_IN_CHAT=0`), optional empty render.
- Evaluation (v3.4)
  - `scripts/eval_metrics.py`: Brier, log-loss, calibration, cohorts.
  - `scripts/eval_runner.py`: scans `eval_data/resolved/*.csv`, writes `eval_runs/<ts>/`.
  - `scripts/eval_ingest.py`: dedup + monthly CSVs; `scripts/eval_weekly.py` wrapper.
  - Auto-commit/push for outputs and optional input data (`TB_EVAL_*`).
- Ops & reliability (v4.3)
  - `autocommit.py`; push gating via env; retries/backoff and degraded-mode in progress.

---

## Testing & validation
- Unit tests for metrics and artifact schema.
- Lightweight test runner: `scripts/run_eval_tests.py`.
- Dry-runs with no-send flags to verify chat formatting and provider behavior.

---

## Latest (v3.1.16 — digest A+ + plain-English)
- Parity: Telegram and Discord both tag `[A+ Setup]` in headers and `(A+)` in Quick Summary.
- Readability: Executive Take, Weekly Plan, Engine Thesis simplified to plain English.
- Safety: Heuristic requires actionable stance, now/near timing, TF alignment, strong/elevated signal, non-high risk.
- Status: Full test suite passing (103/103).

---

## Key env controls
- Messaging safety: `TB_HUMAN_DIGEST`, `TB_NO_TELEGRAM`, `TB_NO_DISCORD`.
- Universe git ops: `TB_UNIVERSE_GIT_AUTOCOMMIT`, `TB_UNIVERSE_GIT_PUSH`.
- Evaluation git ops: `TB_EVAL_GIT_AUTOCOMMIT`, `TB_EVAL_GIT_PUSH`, `TB_EVAL_GIT_INCLUDE_DATA`.
- Polymarket: `TB_POLYMARKET_DEBUG`, `TB_POLYMARKET_SHOW_EMPTY`, `TB_POLYMARKET_NUMBERS_IN_CHAT`.
- Perplexity keys: `PPLX_API_KEY` or rotated `PPLX_API_KEY_1..N`.

---

## Current focus / next
- Reliability ops (v4.3): finalize retries/backoff, degraded-run markers, self-checks.
- Optional calibration plots for evaluation.
- CI/scheduler for weekly evaluation snapshots (optional).

---

## References (code)
- Universe: `scripts/tracer_bullet_universe.py`, `config/universe.yaml`, `universe_runs/`
- Digests: `scripts/tg_sender.py`, `scripts/discord_sender.py`, formatters in `scripts/`
- Polymarket: `providers/polymarket_pplx.py`, `scripts/polymarket_bridge.py`
- Evaluation: `scripts/eval_metrics.py`, `scripts/eval_runner.py`, `scripts/eval_ingest.py`, `scripts/eval_weekly.py`
- Ops: `autocommit.py`
