#!/usr/bin/env bash
# Periodic autocommit script to catch any missed artifacts
# Should be run every 5-10 minutes via cron

cd "$(dirname "$0")/.."

# Commit any missed artifacts
python3 -c "
import autocommit as ac
result = ac.auto_commit_and_push(['runs', 'eval_runs', 'universe_runs', 'bars', 'trader_loop.log', 'trading_agent.log'], extra_message='periodic cleanup', push_enabled=True)
print(f'Periodic autocommit: {result}')
" || echo "Periodic autocommit failed"
