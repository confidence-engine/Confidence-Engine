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

## Crypto signals digest (Discord)
- Dry run (latest universe, no sends):
```
python3 scripts/crypto_signals_digest.py
```

- Print Markdown preview:
```
python3 scripts/crypto_signals_digest.py --print-md
```

- Send to crypto-signals channel via webhook:
```
TB_ENABLE_DISCORD=1 DISCORD_CRYPTO_SIGNALS_WEBHOOK_URL="https://discord.com/api/webhooks/…" \
python3 scripts/crypto_signals_digest.py --send
```

- Seamless autosend via .env (no flags):
```
# .env
TB_ENABLE_DISCORD=1
TB_CRYPTO_DIGEST_AUTOSEND=1
DISCORD_CRYPTO_SIGNALS_WEBHOOK_URL="https://discord.com/api/webhooks/…"

# then simply run
python3 scripts/crypto_signals_digest.py
```

- Behavior notes:
  - Crypto digest mirrors the full universe digest formatting and per-timeframe plans but excludes Polymarket by default.
  - Suppression is controlled via env toggles read by the script: `TB_POLYMARKET_SECTION=0`, `TB_POLYMARKET_SHOW_EMPTY=0`.
  - Universe digests are unchanged and still include Polymarket.

- Specify an explicit universe artifact:
```
python3 scripts/crypto_signals_digest.py --universe universe_runs/universe_YYYYMMDD_HHMMSS_v31.json
```

## Polymarket-only digest
- Safe dry-run (no sends; writes polymarket_digest.md):
```
TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 TB_ENABLE_POLYMARKET=1 \
python3 scripts/polymarket_digest_send.py
```

- Send to a dedicated Discord channel via separate webhook:
```
TB_ENABLE_DISCORD=1 DISCORD_POLYMARKET_WEBHOOK_URL="https://discord.com/api/webhooks/…" \
python3 scripts/polymarket_digest_send.py
```

- Falls back to default `DISCORD_WEBHOOK_URL` if `DISCORD_POLYMARKET_WEBHOOK_URL` is unset.

## Crypto Signals → Alpaca (paper)
- Configure `.env`:
```
ALPACA_API_KEY_ID=...
ALPACA_API_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
TB_TRADER_OFFLINE=1   # preview-only
TB_NO_TRADE=1         # safety gate
```

- Preview-only (no API calls, no orders):
```
TB_TRADER_OFFLINE=1 TB_NO_TRADE=1 \
python3 scripts/crypto_signals_trader.py --tf 1h --max-coins 6 --debug
```

- Paper live (API calls enabled; flip gate to allow orders):
```
TB_TRADER_OFFLINE=0 TB_NO_TRADE=1 \
python3 scripts/crypto_signals_trader.py --tf 1h --max-coins 6 --debug

# When ready to actually place orders, flip TB_NO_TRADE=0
TB_TRADER_OFFLINE=0 TB_NO_TRADE=0 \
python3 scripts/crypto_signals_trader.py --tf 1h --max-coins 6 --debug
```

- Recommended flags for higher-quality trades:
  - `--entry-tolerance-bps 10` — small band around entry to reduce misses
  - `--entry-mid-zone` — use mid of entry zone for trigger checks
  - `--min-rr 2.0` — require minimum risk-reward ratio (plan-based)
  - `--cooldown-sec 3600` — 1h cooldown to avoid rapid re-entry
  - `--order-ttl-min 30` — cancel stale open orders older than 30 minutes
  - `--allow-shorts` — allow short sells; when omitted (default), SELLs require an existing base position > 0 (env: `TB_TRADER_ALLOW_SHORTS`)
  - Example:
  ```
  TB_TRADER_OFFLINE=0 TB_NO_TRADE=0 TB_TRADER_NOTIFY=1 TB_ENABLE_DISCORD=1 \
  python3 scripts/crypto_signals_trader.py \
    --tf 4h --symbols BTC/USD,ETH/USD \
    --entry-tolerance-bps 10 --entry-mid-zone --min-rr 2.0 \
    --cooldown-sec 3600 --order-ttl-min 30 --debug
  ```

- Behavior:
  - Builds digest via `scripts/crypto_signals_digest.py` internals and extracts per-TF plans.
  - Maps plan entries/invalidation/targets to bracket orders sized by risk fraction.
  - Defaults: timescale `1h`, risk 0.5% equity; configurable via flags/env.
  - Duplicate protections: checks existing positions and open orders per symbol/side.
  - Cooldown/state: persists `state/crypto_trader_state.json` to avoid rapid re-entry (configurable via `TB_TRADER_COOLDOWN_SEC`).
  - Live price trigger: requires current price to cross entry in the direction of trade.
  - Position-aware SELL gate: on spot crypto, SELL orders are skipped when there is no base position and `--allow-shorts` is not set (or `TB_TRADER_ALLOW_SHORTS=0`). Discord/journal note shows `skipped:no_position_for_sell`.
  - Final SELL safety clamp: before submit, SELL qty is capped to available base position; if position is zero, trade is skipped. Journal `note` includes `qty_capped_to_position(<qty>)` when clamped.
  - Shorts support: Alpaca spot crypto does not support shorting. Keep `--allow-shorts` off unless trading on a venue that supports crypto shorts.

- Continuous scheduling (loop, every N seconds):
```
TB_TRADER_OFFLINE=0 TB_NO_TRADE=1 \
python3 scripts/crypto_signals_trader.py --tf 1h --loop --interval-sec 60 --debug
```

  - With quality/safety flags and 5-minute cadence:
  ```
  TB_TRADER_OFFLINE=0 TB_NO_TRADE=0 TB_TRADER_NOTIFY=1 TB_ENABLE_DISCORD=1 \
  python3 scripts/crypto_signals_trader.py \
    --tf 4h --symbols BTC/USD,ETH/USD \
    --entry-tolerance-bps 10 --entry-mid-zone --min-rr 2.0 \
    --cooldown-sec 3600 --order-ttl-min 30 \
    --loop --interval-sec 300 --debug
  ```

### 24/7 background via launchd (macOS)

Run the trader every 5 minutes in the background (no explicit `--loop`), letting the script enforce a 1h cooldown:

1) Create `~/Library/LaunchAgents/com.tracer.crypto-trader.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.tracer.crypto-trader</string>
  <key>WorkingDirectory</key><string>/Users/mouryadamarasing/Documents/Project-Tracer-Bullet</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/env</string><string>python3</string><string>scripts/crypto_signals_trader.py</string>
    <string>--tf</string><string>4h</string>
    <string>--symbols</string><string>BTC/USD,ETH/USD</string>
    <string>--entry-tolerance-bps</string><string>10</string>
    <string>--entry-mid-zone</string>
    <string>--min-rr</string><string>2.0</string>
    <string>--cooldown-sec</string><string>3600</string>
    <string>--order-ttl-min</string><string>30</string>
    <string>--debug</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>TB_TRADER_OFFLINE</key><string>0</string>
    <key>TB_NO_TRADE</key><string>0</string>
    <key>TB_TRADER_NOTIFY</key><string>1</string>
    <key>TB_ENABLE_DISCORD</key><string>1</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>300</integer>
  <key>StandardOutPath</key><string>/Users/mouryadamarasing/Documents/Project-Tracer-Bullet/trader_loop.log</string>
  <key>StandardErrorPath</key><string>/Users/mouryadamarasing/Documents/Project-Tracer-Bullet/trader_loop.err</string>
</dict></plist>
```

2) Load and start:
```
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.tracer.crypto-trader.plist 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tracer.crypto-trader.plist
launchctl enable gui/$(id -u)/com.tracer.crypto-trader
launchctl kickstart -k gui/$(id -u)/com.tracer.crypto-trader
```

3) Monitor:
```
tail -n 120 /Users/mouryadamarasing/Documents/Project-Tracer-Bullet/trader_loop.log
tail -n 50  /Users/mouryadamarasing/Documents/Project-Tracer-Bullet/state/trade_journal.csv
```

- Place-and-cancel test order (paper):
  - Safely validates submission/cancel permissions without filling (deep limit)
  - Adjusts quantity to satisfy Alpaca's minimum notional ($10)
```
python3 - << 'PY'
import os, time, math
from pathlib import Path
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST

root = Path('.').resolve()
load_dotenv(dotenv_path=root / '.env')

key=os.getenv('ALPACA_API_KEY_ID'); sec=os.getenv('ALPACA_API_SECRET_KEY'); url=os.getenv('ALPACA_BASE_URL','https://paper-api.alpaca.markets')
assert key and sec, "Missing ALPACA keys in env"
api=REST(key, sec, base_url=url)

symbol='BTC/USD'
bars = api.get_crypto_bars([symbol], '1Min', limit=5).df
px = float(bars['close'].iloc[-1]) if len(bars)>0 else 50000.0
limit_price = round(px * 0.80, 2)  # deep buy to avoid fill
min_notional = 10.50
qty = math.ceil((min_notional / limit_price) * 1e6) / 1e6  # 6dp precision

print(f"Placing test limit BUY {symbol} qty={qty} @ {limit_price} (px~{px}) notional~${qty*limit_price:.2f}")
order = api.submit_order(
    symbol=symbol,
    side='buy',
    type='limit',
    time_in_force='gtc',
    qty=qty,
    limit_price=limit_price,
)
oid = getattr(order, 'id', None) or getattr(order, 'client_order_id', None)
print("Submitted order_id:", oid)

time.sleep(1.0)
api.cancel_order(oid)
time.sleep(0.5)
try:
    o = api.get_order(oid)
    print("Post-cancel status:", getattr(o, 'status', None))
except Exception as e:
    print("Fetch after cancel errored (ok if purged):", e)
PY
```

Note: Alpaca requires a minimum order notional (~$10). Use a deep limit to avoid accidental fills.

- Live trade notifications to Discord (optional):
  - Configure `.env`:
```
TB_TRADER_NOTIFY=1
DISCORD_TRADER_WEBHOOK_URL=https://discord.com/api/webhooks/<id>/<token>  # live-trades channel webhook
TB_ENABLE_DISCORD=1
```
  - The trader will post intent previews (offline/no-trade) and submissions with concise embeds.
  - Safety: Honors `TB_ENABLE_DISCORD` and requires `TB_TRADER_NOTIFY=1` explicitly.

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

