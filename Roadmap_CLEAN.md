# Tracer Bullet — Roadmap (Clean, Detailed)

A milestone-driven plan from V1 tracer bullet to V7 live capital, with crisp DoD (definition of done), acceptance criteria, risks, and operating cadence.

Last updated: 2025-08-20.

---

## 0) Status Summary
- V1 — Tracer Bullet: DONE
- V2 — Crowd Immunity: DONE
- V3 — Bias Immunity + Sizing: DONE
- V3.1 — Multi-Asset Foundations: DONE
- V3.3 — Evidence Lines + Polymarket (read-only): DONE
- V3.4 — Evaluation Pipeline: IN PROGRESS (runner/metrics shipped; accumulating obs)
- V4.2 — Backtesting & Governance: NOT STARTED
- V4.3 — Reliability Ops Hardening: IN PROGRESS (nightly hit‑rate self‑checks; broadened non‑.py auto‑commit)
- V5 — 24/7 Cloud Agent: NOT STARTED
- V6 — Paper Execution & Risk: NOT STARTED
- V7 — Live Capital: NOT STARTED

---

## 1) Now → Near → Next
- Now — v4.3 Reliability Ops Hardening
  - Nightly self‑checks: hit‑rate trend append + regression compare (env‑tunable)
  - Auto‑commit scope broadened (stage all; unstage *.py) so `runs/*.json` and `universe_runs/metrics.csv` are pushed
  - Retries/backoff, timeouts, schema/digest self‑checks, degraded‑run markers, circuit breakers
  - Acceptance: 3‑day burn‑in; <1% degraded runs; zero crashes
- Near — v3.4 Evaluation Pipeline
  - ≥50 resolved obs; weekly snapshots; Brier/log-loss, calibration bins, cohorts
  - Optional calibration plots; commit/push artifacts by default (env-gated)
- Next — v4.2 Backtesting & Governance
  - Event-ordered replay; point-in-time features; walk-forward with IS/OOS
  - Monthly governance cadence; cohort reporting

---

## 2) Milestones & DoD

### V1 — Tracer Bullet (single-asset loop) [DONE]
- Scope: BTCUSD; Alpaca bars; narrative ingest; FinBERT+LLM polarity; divergence; Telegram preview; SQLite logging.
- DoD: end-to-end decision preview; safe parser rejections; traceable messages.

### V2 — Better Signals (decay, novelty, explainability) [DONE]
- Scope: event-driven triggers; decay; novelty weighting; 10–20 symbols; explainable messages.
- DoD: fewer, higher-quality signals; no duplicate triggers within cooldown.

### V3 — Bias Immunity + Sizing [DONE]
- Scope: confirmation and alignment checks; participation; volatility-aware sizing; caps/floors.
- DoD: improved precision; sizing bounded; readable rationale.

### V3.1 — Multi‑Asset Foundations [DONE]
- Scope: crypto+stocks; trading hours; orchestrator; universe digest.
- DoD: stable multi‑asset runs; digest compiles; ranked top‑N.

### V3.3 — Evidence Lines + Polymarket (read‑only) [DONE]
- Scope: number‑free evidence lines in chat; artifacts keep numbers; PPLX-only Polymarket discovery; internal_prob side-by-side.
- DoD: digest shows alts + evidence; Polymarket section present when quality met.

### V3.4 — Evaluation Pipeline (Polymarket + System) [IN PROGRESS]
- Scope: `scripts/eval_metrics.py`, `eval_runner.py`, `eval_ingest.py`, `eval_weekly.py`; outputs in `eval_runs/<ts>/`.
- Metrics: Brier, log‑loss, calibration bins CSV, cohort win‑rates.
- DoD: ≥50 resolved obs; reproducible metrics; weekly snapshots; optional plots.

### V4.2 — Backtesting & Governance [NEXT]
- Scope: event‑ordered replay; point‑in‑time features; walk‑forward; cohort analytics; governance.
- DoD: leak‑free; thresholds documented from OOS; cohort report produced.

### V4.3 — Reliability Ops Hardening [IN PROGRESS]
- Scope: retries/backoff, timeouts, schema/digest self‑checks, degraded markers, circuit breakers; git auto‑push polish.
- DoD: 3‑day burn‑in; <1% degraded; zero crashes; clear logs.

### V5 — 24/7 Cloud Agent [LATER]
- Scope: scheduled GH Actions cadence; secrets; monitoring/rollback; include Polymarket if stable.
- DoD: multi‑week stable schedule; safe pause/rollback.

### V6 — Paper Execution & Risk Controls [LATER]
- Scope: dry‑run execution sim; portfolio caps; per‑asset limits; kill‑switches.
- DoD: stable dry‑run; guardrails verified.

### V7 — Live Capital (Small, Guarded) [LATER]
- Scope: tiny notional; strict loss limits; anomaly alerts; rollback rehearsed; Polymarket exec only if ≥6mo strong eval.
- DoD: safe, audited trial; incident drills passed.

---

## 3) Cross‑Cutting Workstreams (Ongoing)
- Docs: keep README/roadmap/runbook/schema digest spec current.
- Observability: metrics per run; weekly eval outputs; degraded markers.
- Safety: schema versioning; graceful degradation; provider circuit breakers; key rotation.
- Governance: version tags in artifacts; monthly param review; quarterly regime assessments.

---

## 4) Metrics & Acceptance (global)
- Reliability: crash‑free runs; <1% degraded; retried tasks succeed within N attempts.
- Quality: improving Brier/log‑loss; calibration residuals centered; cohort win‑rates stable.
- Ops: auto‑commit/push success rates; zero lingering untracked artifacts.
- Delivery: digest send success (when enabled); no oversized message failures (splitter).

---

## 5) Operating Cadence
- Daily: agent runs; dev logs updated; triage errors; light wiki edits.
- Weekly: review signals/cohorts; run `eval_weekly.py`; snapshot metrics.
- Monthly: governance; walk‑forward update; universe refresh.
- Quarterly: regime analysis; scale/retire cohorts; source/tactics research.

---

## 6) Dependencies & Artefacts
- Providers: Alpaca market data; PPLX (Perplexity) for structured news; Polymarket via PPLX.
- Artefacts: `universe_runs/`, `runs/`, `eval_runs/`, enriched JSON/CSV with evidence lines and polymarket arrays.
- Git Ops: `autocommit.py`; `TB_*_GIT_AUTOCOMMIT`, `TB_*_GIT_PUSH`, include data flags.

---

## 7) Risks & Mitigations
- Provider outages → retries/backoff, key rotation, degraded‑run markers.
- Schema drift → strict validators; explicit versioning; repair‑then‑reject.
- Overfitting in backtests → point‑in‑time features; walk‑forward IS/OOS; governance cadence.
- Artifact bloat → pruning/rotation or archival strategy (later).

---

## 8) Immediate Next Actions
1. Finish v4.3 self‑checks + retries/backoff + auto‑push polish.
2. Accumulate resolved observations; ship optional calibration plots.
3. Outline backtesting harness interfaces; define point‑in‑time feature extracts.

---

## 9) Pointers (code)
- Universe: `scripts/tracer_bullet_universe.py`, `config/universe.yaml`, `universe_runs/`
- Polymarket: `providers/polymarket_pplx.py`, `scripts/polymarket_bridge.py`
- Evaluation: `scripts/eval_metrics.py`, `scripts/eval_runner.py`, `scripts/eval_ingest.py`, `scripts/eval_weekly.py`
- Ops: `autocommit.py`, `.env.example`
