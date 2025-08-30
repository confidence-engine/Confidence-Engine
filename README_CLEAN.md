# Confidence Engine — Agile Divergence Engine

A pragmatic research/trading agent that exploits the gap between narrative (news) and tape (price) with discipline, explainability, and continuous evaluation.

---

## 1) What it does
- Detects divergences between structured news momentum and multi‑timescale price action.
- Produces rich per‑asset artifacts with narratives, evidence lines, confidence, and Polymarket mapping.
- Delivers optional digest messages to Telegram and Discord (env‑gated, safe split/chunking).
- Continuously evaluates probability quality (Brier, log‑loss, calibration) and auto‑publishes CSVs.

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

---

## Important disclaimers

- This is a research/learning project. Not financial advice. Markets are volatile; use at your own risk.
- Experimental/testing phase: results may vary and won’t be perfectly consistent while guardrails are tuned.

---

## Tech stack used to build

- Perplexity API for LLM-based narrative synthesis (source‑tagged evidence lines)
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

---

## 12) License
See `LICENSE`.
