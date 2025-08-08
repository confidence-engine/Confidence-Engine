# Project Tracer Bullet — Architecture

## Current Baseline (V1.0a)
- Flat files in project root:
  - config.py — loads .env settings
  - alpaca.py — data fetch (crypto bars, headlines)
  - finbert.py — sentiment scoring (FinBERT)
  - price.py — RSI/MACD/MA-slope composite
  - narrative_dev.py — narrative stub + BTC relevance (semantic/keyword)
  - narrative_analysis.py — blend + time decay
  - divergence.py — divergence + reason codes
  - explain.py — human-readable labels for scores
  - time_utils.py — time helpers
  - db.py — SQLite persistence
  - schema.sql — DB schema
  - tracer_bullet.py — orchestrator (fetch → analyze → persist → print)
  - view_runs.py — list and inspect saved runs
  - notifier.py — Telegram send (optional)
  - export.py — JSON export and bar snapshot persistence

## Data Flow
1) Fetch bars/headlines (alpaca.py)
2) Filter and summarize headlines (narrative_dev.py with relevance)
3) Score sentiment (finbert.py, relevant-only)
4) Compute price score and volume z (price.py)
5) Blend and decay narrative (narrative_analysis.py)
6) Compute divergence and reason (divergence.py)
7) Explain labels and summaries (explain.py)
8) Persist run (db.py; runs table)
9) Optional: export JSON and bar snapshots (export.py)
10) Optional: notify Telegram (notifier.py)

## Persistence
- SQLite: tracer.db (runs table)
- Optional JSON: runs/<run_id>.json
- Optional bars: bars/<run_id>.csv

## Environments
- .env controls keys and thresholds; no quotes for values.
