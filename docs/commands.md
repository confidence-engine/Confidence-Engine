# Tracer Bullet — Important Commands

Safe profile flags (recommended during local testing):
- TB_NO_TELEGRAM=1 TB_NO_DISCORD=1  # prevent external sends
- TB_UNIVERSE_GIT_AUTOCOMMIT=0 TB_UNIVERSE_GIT_PUSH=0  # do not auto-commit/push
- TB_EVAL_GIT_AUTOCOMMIT=0 TB_EVAL_GIT_PUSH=0

## Universe & digests
- Run universe (no sends):
```
TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 \
python3 scripts/tracer_bullet_universe.py --config config/universe.yaml --top 10
```

## Evaluation and accuracy
- Update hit-rate trend + write daily summary and failures (local safe):
```
TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 \
python3 scripts/asset_hit_rate.py --runs_dir universe_runs --bars_dir bars --runs_map_dir runs --debug \
  --failures_csv eval_runs/hit_rate_failures.csv --markdown_out eval_runs/hit_rate_summary.md
```

- Compare today vs previous night (warn-only):
```
TB_HITRATE_REG_THRESH=0.05 python3 scripts/hit_rate_compare.py
```

- Plot hit-rate trend to PNG:
```
python3 scripts/plot_hit_rate_trend.py --csv eval_runs/hit_rate_trend.csv --out eval_runs/hit_rate_trend.png
```

- Where to look:
  - Trend CSV: `eval_runs/hit_rate_trend.csv`
  - Summary: `eval_runs/hit_rate_summary.md`
  - Failures: `eval_runs/hit_rate_failures.csv`
  - Plot: `eval_runs/hit_rate_trend.png`

## Tests
- Run test suite (safe profile):
```
TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 TB_UNIVERSE_GIT_AUTOCOMMIT=0 TB_UNIVERSE_GIT_PUSH=0 TB_EVAL_GIT_AUTOCOMMIT=0 TB_EVAL_GIT_PUSH=0 \
python3 -m pytest -q
```

## Nightly CI behavior
- Hit-rate trend appended, regression compared (warn-only).
- Auto-commit policy: stage all, unstage `*.py`, commit/push only artifacts (JSON/CSV/MD/YAML).
- See workflow: `.github/workflows/safe_qa_nightly.yml`.

## Tunables (env vars)
- TB_HITRATE_SIDEWAYS_EPS — sideways band for scoring
- TB_HITRATE_W_1H, TB_HITRATE_W_4H, TB_HITRATE_W_1D — horizon weights
- TB_HITRATE_REG_THRESH — nightly regression warning threshold
- TB_NO_TELEGRAM, TB_NO_DISCORD — disable sends during safe runs
- TB_UNIVERSE_GIT_AUTOCOMMIT/PUSH, TB_EVAL_GIT_AUTOCOMMIT/PUSH — control git ops
