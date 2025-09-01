#!/usr/bin/env bash
set -euo pipefail
# Start the hybrid agent autonomous loop (nohup) with auto-apply and robustness gates.
# Safe defaults can be overridden via environment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Preflight: ensure promoted_params.json is fresh; auto-run weekly script if stale/missing
PROMOTED_JSON="config/promoted_params.json"
MAX_PROMOTED_AGE_DAYS="${TB_START_MAX_PROMOTED_AGE_DAYS:-8}"
NEED_REFRESH=0
if [ ! -f "$PROMOTED_JSON" ]; then
  NEED_REFRESH=1
else
  NOW_TS=$(date +%s)
  pm_mtime=$(stat -f %m "$PROMOTED_JSON" 2>/dev/null || stat -c %Y "$PROMOTED_JSON" 2>/dev/null || echo 0)
  pm_age_days=$(( (NOW_TS - pm_mtime) / 86400 ))
  if [ "$pm_age_days" -gt "$MAX_PROMOTED_AGE_DAYS" ]; then
    NEED_REFRESH=1
  fi
fi
if [ "$NEED_REFRESH" = "1" ]; then
  echo "[start_hybrid_loop] promoted_params stale/missing; running weekly_propose_canary.sh before launch" | tee -a trader_loop.log
  bash scripts/weekly_propose_canary.sh || true
fi

nohup bash -lc '
set -a; [ -f .env ] && source .env; set +a
export PYTHONPATH="${PYTHONPATH:-$PWD}"

# Notifications, heartbeats, audit
export TB_TRADER_NOTIFY=${TB_TRADER_NOTIFY:-1}
export TB_TRADER_NOTIFY_HEARTBEAT=${TB_TRADER_NOTIFY_HEARTBEAT:-1}
export TB_HEARTBEAT_EVERY_N=${TB_HEARTBEAT_EVERY_N:-12}
export TB_AUDIT=${TB_AUDIT:-1}

# Telegram/Discord
export TB_NO_TELEGRAM=${TB_NO_TELEGRAM:-0}
export TB_ENABLE_DISCORD=${TB_ENABLE_DISCORD:-1}

# Live auto-apply (auto-learn promoted params)
export TB_AUTO_APPLY_ENABLED=${TB_AUTO_APPLY_ENABLED:-1}
export TB_AUTO_APPLY_KILL=${TB_AUTO_APPLY_KILL:-0}

# Robustness gates
export TB_USE_ATR_FILTER=${TB_USE_ATR_FILTER:-1}
export TB_ATR_LEN=${TB_ATR_LEN:-14}
export TB_ATR_MIN_PCT=${TB_ATR_MIN_PCT:-0.002}
export TB_ATR_MAX_PCT=${TB_ATR_MAX_PCT:-0.10}
export TB_USE_HTF_REGIME=${TB_USE_HTF_REGIME:-1}
export TB_HTF_EMA_LEN=${TB_HTF_EMA_LEN:-200}

# ML gate + retraining
export TB_USE_ML_GATE=${TB_USE_ML_GATE:-1}
export TB_ML_GATE_MODEL_PATH=${TB_ML_GATE_MODEL_PATH:-eval_runs/ml/latest/model.pt}
export TB_ML_FEATURES_PATH=${TB_ML_FEATURES_PATH:-eval_runs/ml/latest/features.csv}
export TB_ML_GATE_MIN_PROB=${TB_ML_GATE_MIN_PROB:-0.5}
export TB_ML_GATE_SOFT=${TB_ML_GATE_SOFT:-1}
export TB_ML_RETRAIN_EVERY_SEC=${TB_ML_RETRAIN_EVERY_SEC:-3600}

# Live trading defaults (set to 1 for dry run)
export TB_TRADER_OFFLINE=${TB_TRADER_OFFLINE:-0}
export TB_NO_TRADE=${TB_NO_TRADE:-0}

# Enhanced Multi-Asset Trading (NEW FEATURES)
export TB_MULTI_ASSET=${TB_MULTI_ASSET:-1}
export TB_ASSET_LIST=${TB_ASSET_LIST:-"BTC/USD,ETH/USD,SOL/USD,LINK/USD,LTC/USD,BCH/USD,UNI/USD,AAVE/USD,AVAX/USD,DOT/USD,MATIC/USD,MKR/USD,COMP/USD,YFI/USD,CRV/USD,SNX/USD,SUSHI/USD,XTZ/USD,GRT/USD"}
export TB_USE_ENHANCED_RISK=${TB_USE_ENHANCED_RISK:-1}
export TB_USE_KELLY_SIZING=${TB_USE_KELLY_SIZING:-1}
export TB_USE_REGIME_DETECTION=${TB_USE_REGIME_DETECTION:-1}
export TB_USE_ENSEMBLE_ML=${TB_USE_ENSEMBLE_ML:-0}
export TB_USE_ADAPTIVE_STRATEGY=${TB_USE_ADAPTIVE_STRATEGY:-1}
export TB_MAX_POSITIONS=${TB_MAX_POSITIONS:-3}
export TB_MAX_CORRELATION=${TB_MAX_CORRELATION:-0.7}
export TB_PORTFOLIO_VAR_LIMIT=${TB_PORTFOLIO_VAR_LIMIT:-0.02}

# Auto-commit artifacts (never commits .py)
export TB_AUTOCOMMIT_ARTIFACTS=${TB_AUTOCOMMIT_ARTIFACTS:-1}
export TB_AUTOCOMMIT_PUSH=${TB_AUTOCOMMIT_PUSH:-1}

# Seed latest model quietly
python3 scripts/ml_retrainer.py --bars_dir bars --out_root eval_runs/ml --link_dir eval_runs/ml/latest >/dev/null 2>&1 || true

# Periodic retraining
( while true; do
  python3 scripts/ml_retrainer.py --bars_dir bars --out_root eval_runs/ml --link_dir eval_runs/ml/latest || true
  sleep "${TB_ML_RETRAIN_EVERY_SEC}"
done ) &

# Trader loop with exploration windows and epsilon-greedy
while true; do
  # Determine exploration mode for this iteration
  NOW_MIN=$(date +%M)
  EXPLORATION_WINDOW=0
  if [ "$NOW_MIN" -ge 10 ] && [ "$NOW_MIN" -lt 20 ]; then
    EXPLORATION_WINDOW=1
  fi

  # 10% epsilon-greedy by default (configurable via TB_EPSILON_PCT)
  R=$((RANDOM % 100))
  EPS=0
  if [ "$R" -lt "${TB_EPSILON_PCT:-10}" ]; then
    EPS=1
  fi

  # Base thresholds (day-trading defaults can be overridden by env)
  BASE_PROB="${TB_ML_GATE_MIN_PROB:-0.35}"
  BASE_ATR="${TB_ATR_MIN_PCT:-0.001}"

  PROB="$BASE_PROB"
  ATR="$BASE_ATR"
  MODE="normal"

  # Exploration window relax
  if [ "$EXPLORATION_WINDOW" = "1" ]; then
    PROB="${TB_EXP_PROB:-0.26}"
    ATR="${TB_EXP_ATR:-0.0007}"
    MODE="window"
  fi

  # Epsilon-greedy relax (overrides window)
  if [ "$EPS" = "1" ]; then
    PROB="${TB_EPS_PROB:-0.22}"
    ATR="${TB_EPS_ATR:-0.0005}"
    MODE="epsilon"
  fi

  # Enforce a minimum ML threshold floor for safety during exploration
  FLOOR="${TB_ML_PROB_FLOOR:-0.25}"
  awk_cmp=$(awk -v a="$PROB" -v b="$FLOOR" "BEGIN{print (a<b)?1:0}")
  if [ "$awk_cmp" = "1" ]; then
    PROB="$FLOOR"
  fi

  # Minimal sizing during exploration to cap risk
  if [ "$MODE" = "window" ] || [ "$MODE" = "epsilon" ]; then
    export TB_SIZE_MIN_R="${TB_SIZE_MIN_R_EXP:-0.05}"
    export TB_SIZE_MAX_R="${TB_SIZE_MAX_R_EXP:-0.15}"
  else
    # Clear exploration overrides (fall back to .env or defaults inside sizing)
    unset TB_SIZE_MIN_R || true
    unset TB_SIZE_MAX_R || true
  fi

  echo "[start_hybrid_loop] gate PROB=$PROB ATR=$ATR mode=$MODE size_min_R=${TB_SIZE_MIN_R:-def} size_max_R=${TB_SIZE_MAX_R:-def}" >> trader_loop.log

  TB_GATE_MODE="$MODE" TB_ML_GATE_MIN_PROB="$PROB" TB_ATR_MIN_PCT="$ATR" python3 scripts/hybrid_crypto_trader.py || true
  sleep 60
done
' > trader_loop.log 2> trader_loop.err & disown

echo "[start_hybrid_loop] launched at $(date -u +%F_%T)" >> trader_loop.log
