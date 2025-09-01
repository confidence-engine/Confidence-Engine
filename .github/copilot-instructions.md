# Confidence Engine — AI Coding Assistant Instructions

## Project Overview
This is a hybrid crypto trading agent that detects divergences between market narrative (news sentiment) and price action. It autonomously ingests multi-source headlines, filters for asset relevance, computes robust sentiment scores, evaluates divergence signals, and executes paper/live trades via Alpaca API. Key philosophy: "Tracer Bullet" — get end-to-end working first, then add complexity.

## Key Architectural Patterns

### Autonomous Loop with ML Gates
- **Main Loop**: `scripts/hybrid_crypto_trader.py` runs periodic scans with EMA signals, sentiment via Perplexity, ML probability gates, and bracket orders.
- **ML Gate**: Uses PyTorch model from `eval_runs/ml/latest/model.pt` to gate trades; features built from 15m bars (ret_1, ema12_slope, vol_chg, cross_up).
- **State Management**: JSON state files in `state/` track position, cooldowns, PnL; reconciled with Alpaca positions on startup.
- **Auto-Commit**: `autocommit.py` stages/commits artifacts (runs/, bars/, logs) but never code; controlled by `TB_AUTOCOMMIT_*` env vars.

### Multi-Source Data Ingestion
- **Headlines**: Perplexity Pro API (`pplx_fetcher.py`) with key rotation (`PPLX_API_KEY_1..N`), CoinDesk RSS (`coindesk_rss.py`), Alpaca news.
- **Deduplication**: `dedupe_utils.py` normalizes text, preserves originals.
- **Relevance Filtering**: `narrative_dev.py` uses semantic similarity against BTC topic; threshold ~0.42.
- **Robust Sentiment**: `sentiment_utils.py` applies MAD outlier removal + 10% trimmed mean on FinBERT scores.

### Divergence Computation
- **Narrative Score**: LLM polarity + FinBERT blend, decayed exponentially (half-life 90min).
- **Price Score**: RSI, EMA slope, MACD histogram, volume Z-score → normalized [-1,1].
- **Divergence**: `z(narrative) - z(price)`; adaptive trigger based on volume participation.
- **Confirmation Checks**: Penalize confidence for contradictions (price vs narrative, low volume, timescale misalignment).

### Notification & Export
- **Channels**: Telegram (`telegram_bot.py`) and Discord webhooks; number-free chat, full details in artifacts.
- **Artifacts**: JSON payloads in `runs/`, bars CSV in `bars/`, accepted headlines TXT; auto-committed.
- **Provenance**: Tags headlines by source (perplexity|alpaca|coindesk); captured in `pplx_provenance`.

## Developer Workflows

### Local Development
- **Run Single Scan**: `python3 tracer_bullet.py` — fetches data, computes divergence, prints decision preview, exports artifacts.
- **Debug Sources**: `python3 debug_sources.py` — validates API keys, counts/samples from each provider.
- **Test Auth**: `python3 test_pplx_auth.py` — HTTP 200/401 checks per Perplexity key.
- **Offline Mode**: Set `TB_OFFLINE=1` for deterministic synthetic bars/headlines.

### Autonomous Operation
- **Start Loop**: `bash scripts/start_hybrid_loop.sh` — launches trader + ML retrainer in background.
- **Health Check**: `bash scripts/health_check.sh` — verifies processes, log freshness, promoted params age.
- **Watchdog**: Cron every 2min restarts loop if dead; uses `ps ax | egrep 'hybrid_crypto_trader.py'`.
- **Weekly Tuning**: `scripts/weekly_propose_canary.sh` runs backtests, promotes params to `config/promoted_params.json`.

### Testing & Evaluation
- **Unit Tests**: `pytest` on `tests/`; focus on parsers, sentiment utils, divergence logic.
- **Backtesting**: `scripts/run_backtest.py` with event-ordered replay; leak-free point-in-time features.
- **Evaluation**: `scripts/eval_runner.py` computes Brier/log-loss, calibration bins; ingests resolved data via `eval_ingest.py`.
- **Hit-Rate Checks**: `scripts/asset_hit_rate.py` validates directional accuracy against bars.

## Integration Points

### External APIs
- **Alpaca**: Trading/orders via `alpaca.py`; bars/news via `rest.py`.
- **Perplexity**: Sentiment synthesis via `sonar-pro` model; strict JSON parsing, key rotation.
- **Telegram**: DMs via `telegram_bot.py`; bot token + chat_id in `.env`.
- **Discord**: Embeds via `send_discord_embed()`; webhook URLs in `.env`.

### Internal Communication
- **Env Config**: All settings in `.env`; loaded via `python-dotenv`; `TB_` prefixed vars.
- **Retry Logic**: `retry_call()` with exponential backoff; status codes [429,500-599].
- **Logging**: Structured to `trading_agent.log`, `trader_loop.log`; includes run IDs, timestamps.
- **Circuit Breakers**: Degrade gracefully on API failures; pause narrative calls if error rate > threshold.

## Project-Specific Conventions

### Naming & Structure
- **Files**: Snake_case Python files; `scripts/` for executables, `providers/` for adapters.
- **Env Vars**: `TB_` prefix (e.g., `TB_NO_TRADE=1` for paper mode); numbered keys for rotation.
- **Directories**: `runs/` for JSON artifacts, `bars/` for CSVs, `state/` for trader state, `eval_runs/` for metrics.
- **Schemas**: Typed dataclasses for payloads; JSON dumps with `default=str` for timestamps.

### Code Patterns
- **Robust Aggregation**: Use `drop_outliers()` + `trimmed_mean()` for sentiment scores.
- **Feature Building**: Align to penultimate bar for training consistency (e.g., `df.iloc[-2]`).
- **State Reconciliation**: Always reconcile JSON state with broker positions on startup.
- **Idempotent Orders**: Unique client_order_ids to avoid duplicates on retry.
- **Number-Free Chat**: Hide numbers in notifications; keep in artifacts for audit.

### Error Handling
- **Graceful Degradation**: If narrative fails, fall back to TA-only mode.
- **Validation**: Strict JSON schemas; attempt repair, then reject.
- **Alerts**: Telegram/Discord on critical failures; circuit-breaker events.

Reference key files: `tracer_bullet.py` (orchestration), `config.py` (env loading), `autocommit.py` (artifact management), `scripts/hybrid_crypto_trader.py` (live loop).
