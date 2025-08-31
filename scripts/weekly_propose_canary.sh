#!/usr/bin/env bash
set -euo pipefail

# Weekly propose + canary runner (safe, offline by default)
# - Runs the offline auto-tuner to propose parameters under eval_runs/auto_tuner/<ts>/
# - Runs the guarded canary manager to paper-trade and, if thresholds pass, promote to config/promoted_params.json
# - Does NOT enable live auto-apply; live agent must opt-in via TB_AUTO_APPLY_ENABLED=1.
# - Commits artifacts (non-code) via autocommit.py when available.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Safe env defaults
export TB_NO_TELEGRAM=${TB_NO_TELEGRAM:-1}
export TB_NO_DISCORD=${TB_NO_DISCORD:-1}
export TB_TRADER_OFFLINE=${TB_TRADER_OFFLINE:-1}
export TB_NO_TRADE=${TB_NO_TRADE:-1}
export TB_AUTOCOMMIT_ARTIFACTS=${TB_AUTOCOMMIT_ARTIFACTS:-1}
export TB_AUTOCOMMIT_PUSH=${TB_AUTOCOMMIT_PUSH:-1}
# Explicitly do NOT enable auto-apply from this script
export TB_AUTO_APPLY_ENABLED=${TB_AUTO_APPLY_ENABLED:-0}
export TB_AUTO_APPLY_KILL=${TB_AUTO_APPLY_KILL:-0}

TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="eval_runs/weekly/${TS}"
mkdir -p "$LOG_DIR"

# 1) Offline auto-tuner with conservative guardrails
# Adjust flags here if you want different min trades or max dd.
{
  echo "[weekly] running auto_tuner.py @ $TS"
  python3 scripts/auto_tuner.py \
    --bars_dir bars \
    --out_root eval_runs/backtests \
    --proposal_min_trades 20 \
    --proposal_max_dd 0.004
} | tee -a "$LOG_DIR/auto_tuner.log"

# 2) Canary manager — guarded paper canary + promotion if pass
{
  echo "[weekly] running canary_manager.py @ $TS"
  python3 scripts/canary_manager.py
} | tee -a "$LOG_DIR/canary_manager.log"

# 2b) Refresh backtest aggregate rollups so dashboards stay current
{
  echo "[weekly] refreshing backtest aggregates @ $TS"
  python3 scripts/backtest_aggregate.py --out_root eval_runs/backtests || python3 scripts/backtest_aggregate.py || true
} | tee -a "$LOG_DIR/backtest_aggregate.log"

# 3) Auto-commit artifacts only (never code)
if [[ "${TB_AUTOCOMMIT_ARTIFACTS}" == "1" ]]; then
  python3 - <<'PY'
import autocommit
paths = [
  'eval_runs/',
  'runs/',
  'universe_runs/',
  'config/promoted_params.json',
  'Dev_logs.md',
]
print('[autocommit] attempting...')
print(autocommit.auto_commit_and_push(paths, extra_message='weekly propose+canary', push_enabled=True))
PY
fi

echo "[weekly] done @ $TS" | tee -a "$LOG_DIR/weekly.log"
