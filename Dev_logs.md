## [v1-hardening] - 2025-08-10
- Preflight/health: scripts/preflight.py creates runs/ and bars/, checks Telegram reachability.
- CLI/logging: scripts/run.py with flags (--symbol, --lookback, --no-telegram, --debug, --health); centralized logging with ISO timestamps.
- Telegram robustness: plain-text default, truncation to 4000 chars, graceful 200/400/429 handling, TB_NO_TELEGRAM honored.
- Tests/CI: unit tests for divergence, telegram formatting, payload schema, dir checks; GitHub Actions for lint and tests.
- Retention: prune runs/ and bars/ to last N files via TB_ARTIFACTS_KEEP (default 500).
- Docs: README updated; .env.example added.

## [v2-crowd] - 2025-08-10
- Source Diversity Engine:
  - Confidence shaping from unique sources; echo penalty on skew.
  - Payload: source_diversity {unique, top_source_share, counts, adjustment}; confidence capped at 0.75.
- Cascade/HYPE Detector:
  - Repetition via simple token overlap; quant confirmation via price_move_pct and max_volume_z.
  - Tag HYPE_ONLY applies confidence_delta -0.03; payload cascade_detector {...}.
- Contrarian Viewport:
  - Tag POTENTIAL_CROWD_MISTAKE under extreme narrative, low divergence, flat price; informational only.
- Tests: added tests for diversity, cascade, contrarian; total suite passing.

## [v3-bias-sizing] - 2025-08-10
- Timescale scoring:
  - short/mid/long tails from 1-min bars; divergence per horizon; combined weighted divergence; alignment flag.
  - Env weights TB_TS_W_SHORT/MID/LONG with renormalization.
- Negative-confirmation checks:
  - price_vs_narrative, volume_support, timescale_alignment; penalties summed and clamped by TB_CONF_PENALTY_MIN.
  - Payload: confirmation_checks[], confirmation_penalty.
- Position sizing (informational):
  - map confidence to target_R via floors/caps; optional volatility normalization; payload.position_sizing.
- Telegram: appended timescales, confirmation penalty, and sizing lines when applicable.
- Tests: added tests for timescales, confirmation, sizing; all passing.

## [v3.1-multi-asset] - 2025-08-11
- Multi-asset universe support:
  - config/universe.yaml for crypto and stock symbols
  - symbol_utils.py for normalization and type detection
  - trading_hours.py for market hours awareness (RTH/CLOSED/24x7)
  - bars_stock.py adapter with stub data support
- Universe orchestrator:
  - scripts/scan_universe.py for multi-symbol analysis
  - ranking by divergence strength and confidence
  - digest_utils.py for formatted summaries
  - universe_runs/ output with timestamped results
- Payload additions:
  - symbol_type (crypto/stock/unknown)
  - market_hours_state (RTH/CLOSED/24x7)
- Tests: comprehensive test suite for all new components.
- Non-breaking: single-symbol scripts/run.py remains fully functional.
- Auto-commit and mirroring:
  - TB_UNIVERSE_MIRROR_TO_RUNS=1: copy universe files to runs/
  - TB_UNIVERSE_GIT_AUTOCOMMIT=1: git add and commit universe results
  - TB_UNIVERSE_GIT_PUSH=1: git push after auto-commit (requires AUTOCOMMIT=1)
  - All git operations wrapped in try/except with clear logging
  - Defaults: all flags off to avoid surprise commits

# Project Tracer Bullet: Development Log

This log will be updated at the end of each development session to track our progress.

Understood. Ending our session.

This concludes our entire strategic brainstorming and project definition phase. You have successfully navigated an extremely complex series of challenges and, through your sharp insights and diligent research, we have forged a definitive, professional-grade master plan.

-----

### **EOD Summary (Friday, August 8, 2025)**

**Accomplished (Project Definition Phase):**

  * We completed a full, brutally honest analysis of our previous project attempts, identifying unreliable APIs as the core bottleneck.
  * We successfully pivoted to a new, more robust project: the **"Quantitative Alpha Engine."**
  * We conducted deep, iterative research to select a professional, stable platform (**Alpaca**) that meets all of our "no compromise" requirements.
  * We architected a definitive, hybrid intelligence strategy ("Narrative vs. Price Momentum") with a clear competitive edge.
  * We finalized a granular, milestone-based roadmap using the **"Tracer Bullet" Development** methodology.

**Pending:**

  * Our next session will officially begin **Phase 0: Foundation Setup** for our new project.

-----

### **README & Documentation Update**

When you return, you can commit all the documentation files we created. Here is a final, consolidated version for your records.

**Commit Message:**
`docs: Finalize official project charter, roadmap, and wiki`

**Final `README.md` Content:**

```markdown
# Project: Tracer Bullet
> **Product Name:** The Quantitative Alpha Engine
> **Status:** Phase 0 - Foundation Setup

## Mission Statement
To build a professional-grade, autonomous trading agent that finds and exploits short-term, sentiment-driven inefficiencies in the stock and crypto markets. The primary goal is the deep acquisition of elite skills in AI development, data engineering, and quantitative finance.
```

**Final `dev_log.md` First Entry:**

```markdown
# Project Tracer Bullet: Development Log

### **Phase 0: Strategy & Planning (`COMPLETED`)**
- **Date:** August 8, 2025
- **Summary:** Completed a multi-day strategic brainstorming phase. After analyzing the failures of previous API-dependent approaches, we have pivoted to a definitive, "no compromise" project: The "Quantitative Alpha Engine." The project will be built on the professional-grade Alpaca API, using a "Tracer Bullet" development methodology. The final project charter, roadmap, and tech stack have been approved.
```

*(The content for `roadmap.md` and `knowledge_wiki.md` is based on my last comprehensive response and is ready for your commit.)*

-----

### Test Question

According to our new "Tracer Bullet" roadmap, what is the single, simple objective of our very first coding script, `tracer_bullet.py`?

---------------------------
# Dev log summary (2025-08-09)

- Added multi-source ingest: Perplexity Pro API with key rotation; CoinDesk RSS.
- Implemented robust FinBERT sentiment: MAD outlier removal + trimmed mean on relevant-only.
- Enriched BTC topic; tuned relevance gating; added keyword fallback.
- Narrative built from accepted headlines; conservative confidence fallback.
- Provenance tagging for accepted; persisted accepted headlines to runs/<id>_accepted.txt.
- Adaptive divergence trigger based on volume Z.
- Debug utilities: inspect_env_pplx, test_pplx_auth, debug_sources.
- Console output improvements: accepted sources/score list, relevance top-5, clear decision preview.
- Auto-commit of artifacts retained; JSON includes robust sentiment details and Perplexity provenance.
## Post-initial-commit updates (Agent V1.1)
- Integrated per-source weighted relevance; using weighted score for acceptance thresholding.
- Enforced Perplexity recency filter to "day" for fresher coverage; retained rotation.
- Replaced process-focused preview with alpha-first summary and actionable next steps.
- Wired Telegram push: auto-sends alpha summary with top evidence to configured chat.
- Restored summary/detail in payload for DB schema alignment; ensured JSON carries alpha_* fields and weighted relevance details.

# Dev Log — 2025-08-09

- Integrated per-source weighted relevance and persisted raw/weighted scores.
- Enforced Perplexity day-recency; added provenance.
- Replaced process-centric text with alpha-first summary and actionable next steps.
- Restored payload summary/detail for DB schema alignment.
- Telegram DM wired: end-of-run message with alpha, what flips to action, and top-3 evidence.
- Added catalyst heuristic to Telegram formatter.

Here are copy-ready doc updates reflecting the recent V1 hardening work (Prompts 1–4) and outcome. Use these to update README, CHANGELOG, RUNBOOK, and .env.example.

1) README sections

- README: Introduction (already aligned with Tracer Bullet mission)
Use only if you haven’t added the mission yet.

Title: Tracer Bullet — The Agile Divergence Engine for Crypto Alpha
Summary:
- Mission: Exploit emotional–math gaps with perfect discipline using a divergence engine (story vs price).
- Not a sentiment bot: trades the reaction gap, not raw mood.
- Architecture: Hybrid Oracle (price/volume) + Psychologist (narrative).
- Outputs: Actionable signals with evidence via Telegram; auditable payloads for backtesting.

- README: Quick start (updated)
Requirements:
- Python 3.11+
- pip install -r requirements.txt
- Telegram bot token and chat ID

Setup:
```bash
cp .env.example .env
# Fill TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (optional for no-telegram runs)
python3 scripts/preflight.py
```

Run:
```bash
# Health check
python3 scripts/run.py --health

# Run without Telegram (safe)
python3 scripts/run.py --symbol BTC/USD --lookback 240 --no-telegram

# Debug logs
python3 scripts/run.py --symbol BTC/USD --lookback 120 --no-telegram --debug
```

Telegram test (optional):
```bash
python3 - << 'PY'
import os, requests
from dotenv import load_dotenv
load_dotenv(".env")
t=os.getenv("TELEGRAM_BOT_TOKEN"); c=os.getenv("TELEGRAM_CHAT_ID")
assert t and c, "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
r=requests.post(f"https://api.telegram.org/bot{t}/sendMessage",
                json={"chat_id": c, "text": "Tracer Bullet V1 hardening OK"}, timeout=10)
print(r.status_code, r.text[:160])
PY
```

- README: Configuration and overrides (new)
Config precedence:
- CLI flags > process env > .env > defaults.

Common envs:
- TB_SYMBOL_OVERRIDE: e.g., BTC/USD
- TB_LOOKBACK_OVERRIDE: e.g., 180
- TB_NO_TELEGRAM: 1 to disable sending
- TELEGRAM_PARSE_MODE: leave empty for plain text
- TB_ARTIFACTS_KEEP: how many files to keep in runs/ and bars/ (default 500)
- LOG_LEVEL: INFO (default) or DEBUG

Examples:
```bash
TB_NO_TELEGRAM=1 TB_SYMBOL_OVERRIDE=BTC/USD TB_LOOKBACK_OVERRIDE=180 python3 scripts/run.py --debug
```

- README: Artifacts and retention (new)
Artifacts:
- runs/: per-run JSON payloads
- bars/: cached bar CSVs

Retention:
- The pipeline prunes older artifacts automatically, keeping the most recent N files (default 500). Configure via TB_ARTIFACTS_KEEP.

2) .env.example (new or updated)

Create/update .env.example:
```
# Required for Telegram sends (optional if TB_NO_TELEGRAM=1)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Leave empty to avoid Telegram formatting/escaping issues
TELEGRAM_PARSE_MODE=

# Runtime overrides (optional)
TB_SYMBOL_OVERRIDE=
TB_LOOKBACK_OVERRIDE=
TB_NO_TELEGRAM=

# Logging
LOG_LEVEL=INFO

# Artifact retention
TB_ARTIFACTS_KEEP=500
```

3) CHANGELOG (new entries)

Add to CHANGELOG.md:

## [v1-hardening] - 2025-08-10
- Preflight and health checks
  - scripts/preflight.py validates env, creates runs/ and bars/, checks Telegram reachability.
- CLI wrapper and logging
  - scripts/run.py with flags: --symbol, --lookback, --no-telegram, --debug, --health.
  - Central logging (ISO timestamps), quieter third-party logs.
- Telegram robustness
  - Defaults to plain text unless parse_mode provided.
  - Truncates messages to 4000 chars.
  - Clean handling of 200/400/429 and exceptions; honors TB_NO_TELEGRAM.
- Tests and CI
  - Minimal unit tests for divergence, Telegram formatting, payload schema, dir checks, retention; 7 tests passing.
  - GitHub Actions for lint (flake8, black --check) and pytest.
- Artifact retention
  - Retention utility to prune runs/ and bars/ to last N files (default 500), configurable via TB_ARTIFACTS_KEEP.
- Docs
  - README quick start and configuration updated.
  - .env.example added.

4) RUNBOOK.md (operations)

Add RUNBOOK.md:

Title: Tracer Bullet Runbook

Health and preflight:
```bash
python3 scripts/preflight.py
python3 scripts/run.py --health
```

Routine run (safe mode, no Telegram):
```bash
python3 scripts/run.py --symbol BTC/USD --lookback 240 --no-telegram
```

Debug run:
```bash
python3 scripts/run.py --symbol BTC/USD --lookback 120 --no-telegram --debug
```

Overrides:
- CLI takes precedence; otherwise set env:
```bash
TB_NO_TELEGRAM=1 TB_SYMBOL_OVERRIDE=ETH/USD TB_LOOKBACK_OVERRIDE=90 python3 scripts/run.py --debug
```

Artifacts:
- Found in runs/ and bars/. Automatic pruning keeps last N files (TB_ARTIFACTS_KEEP, default 500).

Telegram:
- Defaults to plain text; set TELEGRAM_PARSE_MODE only if needed.
- Errors (400/429) are logged, run continues; TB_NO_TELEGRAM skips send.

Troubleshooting:
- No DM received: ensure you started the bot in Telegram, TELEGRAM_CHAT_ID is correct, TELEGRAM_PARSE_MODE is blank (plain) to avoid formatting rejects.
- Tests:
```bash
python3 -m pytest -q
```

5) CONTRIBUTING.md (light)

Add CONTRIBUTING.md:

- Use Python 3.11, minimal dependencies.
- Keep secrets in .env (never commit).
- Run tests and lint before PR:
```bash
python3 -m pytest -q
flake8 .
black --check .
```
- Respect config precedence and don’t remove payload keys used downstream.
- Prefer pure functions for new scoring logic; add unit tests.

6) Repository file overview (docs snippet)

Add to README or a docs/overview.md:

- scripts/preflight.py: env/folder/network checks
- scripts/run.py: CLI wrapper and health
- logging_utils.py: logging setup
- telegram_bot.py: Telegram delivery (plain default, truncation, robust errors)
- tracer_bullet.py: main pipeline entrypoint
- retention.py: artifact pruning utilities
- tests/: unit tests (7 passing)
- .github/workflows/ci.yml: CI for lint and tests

7) Optional: payload documentation (schema excerpt)

Add docs/payload.md:

- Required keys present in runs/*.json:
  - alpha_summary (str)
  - alpha_next_steps (str)
  - relevance_details (JSON str with accepted[])
  - summary (str), detail (str)
  - divergence_threshold (float)
  - confidence (float)
  - divergence (float)
  - action (str)
- Example access:
```python
import json
d = json.load(open("runs/last.json"))
print(d["alpha_summary"], d["confidence"])
```
Here are copy-ready doc updates reflecting everything you just implemented (V1 hardening + V2 Crowd Immunity modules). Paste into the indicated files.

1) README.md
Title: Tracer Bullet — Agile Divergence Engine for Crypto Alpha

Summary:
- Mission: Exploit story vs price gaps with discipline.
- Architecture: Price/volume oracle + narrative analyzer.
- Outputs: Actionable signal + evidence via Telegram; auditable JSON payloads.

Quick Start:
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
cp .env.example .env   # fill tokens if using Telegram
python3 scripts/preflight.py
```

Run:
```bash
# Health
python3 scripts/run.py --health

# Safe (no Telegram)
python3 scripts/run.py --symbol BTC/USD --lookback 240 --no-telegram

# Debug logs
python3 scripts/run.py --symbol BTC/USD --lookback 120 --no-telegram --debug
```

Telegram test (optional):
```bash
python3 - << 'PY'
import os, requests
from dotenv import load_dotenv
load_dotenv(".env")
t=os.getenv("TELEGRAM_BOT_TOKEN"); c=os.getenv("TELEGRAM_CHAT_ID")
assert t and c, "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
r=requests.post(f"https://api.telegram.org/bot{t}/sendMessage",
                json={"chat_id": c, "text": "Tracer Bullet: setup OK"}, timeout=10)
print(r.status_code, r.text[:160])
PY
```

Configuration and overrides:
- Precedence: CLI flags > process env > .env > defaults.
- Common envs:
  - TB_SYMBOL_OVERRIDE (e.g., BTC/USD)
  - TB_LOOKBACK_OVERRIDE (e.g., 180)
  - TB_NO_TELEGRAM=1 to disable sends
  - TELEGRAM_PARSE_MODE leave empty for plain text
  - TB_ARTIFACTS_KEEP default 500
  - LOG_LEVEL INFO or DEBUG

Example:
```bash
TB_NO_TELEGRAM=1 TB_SYMBOL_OVERRIDE=BTC/USD TB_LOOKBACK_OVERRIDE=180 python3 scripts/run.py --debug
```

Artifacts & retention:
- runs/: per-run JSON payloads
- bars/: cached bar CSVs
- Automatic pruning keeps most recent N files (TB_ARTIFACTS_KEEP, default 500)

V2 Crowd Immunity features:
- Source Diversity Engine:
  - Adjusts confidence based on unique sources and echo-chamber skew
  - Payload: source_diversity {unique, top_source_share, counts, adjustment}
- Cascade/HYPE Detector:
  - Flags repetitive narrative without quant confirmation
  - Payload: cascade_detector {repetition_ratio, price_move_pct, max_volume_z, tag, confidence_delta}
- Contrarian Viewport:
  - Informational tag for potential crowd mistakes under extreme narrative + flat price + low gap
  - Payload: contrarian_viewport "POTENTIAL_CROWD_MISTAKE" or ""

Tests & CI:
```bash
python3 -m pytest -q
# CI runs flake8, black --check, pytest on PR/push
```

2) .env.example
```
# Telegram (optional if TB_NO_TELEGRAM=1)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
TELEGRAM_PARSE_MODE=

# Overrides
TB_SYMBOL_OVERRIDE=
TB_LOOKBACK_OVERRIDE=
TB_NO_TELEGRAM=

# Logging
LOG_LEVEL=INFO

# Retention
TB_ARTIFACTS_KEEP=500
```

3) CHANGELOG.md
## [v1-hardening] - 2025-08-10
- Preflight/health: scripts/preflight.py creates runs/ and bars/, checks Telegram reachability.
- CLI/logging: scripts/run.py with flags (--symbol, --lookback, --no-telegram, --debug, --health); centralized logging with ISO timestamps.
- Telegram robustness: plain-text default, truncation to 4000 chars, graceful 200/400/429 handling, TB_NO_TELEGRAM honored.
- Tests/CI: unit tests for divergence, telegram formatting, payload schema, dir checks; GitHub Actions for lint and tests.
- Retention: prune runs/ and bars/ to last N files via TB_ARTIFACTS_KEEP (default 500).
- Docs: README updated; .env.example added.

## [v2-crowd] - 2025-08-10
- Source Diversity Engine:
  - Confidence shaping from unique sources; echo penalty on skew.
  - Payload: source_diversity {unique, top_source_share, counts, adjustment}; confidence capped at 0.75.
- Cascade/HYPE Detector:
  - Repetition via simple token overlap; quant confirmation via price_move_pct and max_volume_z.
  - Tag HYPE_ONLY applies confidence_delta -0.03; payload cascade_detector {...}.
- Contrarian Viewport:
  - Tag POTENTIAL_CROWD_MISTAKE under extreme narrative, low divergence, flat price; informational only.
- Tests: added tests for diversity, cascade, contrarian; total suite passing.

4) RUNBOOK.md
Health:
```bash
python3 scripts/preflight.py
python3 scripts/run.py --health
```

Routine run (no Telegram):
```bash
python3 scripts/run.py --symbol BTC/USD --lookback 240 --no-telegram
```

Debug:
```bash
python3 scripts/run.py --symbol BTC/USD --lookback 120 --no-telegram --debug
```

Overrides:
```bash
TB_NO_TELEGRAM=1 TB_SYMBOL_OVERRIDE=ETH/USD TB_LOOKBACK_OVERRIDE=90 python3 scripts/run.py --debug
```

Artifacts:
- Auto pruned to TB_ARTIFACTS_KEEP (default 500)

Telegram:
- Defaults to plain text; set TELEGRAM_PARSE_MODE only if you need Markdown/HTML.
- 400/429 handled gracefully; TB_NO_TELEGRAM skips send.

Troubleshooting:
- No DM: DM “Start” to the bot, verify chat ID, ensure parse mode blank.
- Tests:
```bash
python3 -m pytest -q
```

5) CONTRIBUTING.md
- Python 3.11; keep secrets in .env (never commit).
- Run before PR:
```bash
flake8 .
black --check .
python3 -m pytest -q
```
- Respect config precedence; don’t remove payload keys.
- For new features, add pure functions + tests.
- Keep Telegram messages <4000 chars; plain text default.

6) docs/payload.md
Required payload keys:
- alpha_summary, alpha_next_steps
- relevance_details (JSON string with accepted[])
- summary, detail
- divergence_threshold, confidence, divergence, action

V2 additions:
- source_diversity: {unique:int, top_source_share:float, counts:dict, adjustment:float}
- cascade_detector: {repetition_ratio:float, price_move_pct:float, max_volume_z:float, tag:str, confidence_delta:float}
- contrarian_viewport: "POTENTIAL_CROWD_MISTAKE" or ""

Example:
```python
import json
d = json.load(open("runs/latest.json"))
print(d["alpha_summary"], d["confidence"], d.get("source_diversity"))
```

7) PR description template (optional)
Title: V1 Hardening + V2 Crowd Immunity

Summary:
- V1: preflight, CLI, logging, Telegram robustness, tests, CI, retention.
- V2: source diversity, cascade/hype detector, contrarian viewport.

Testing:
- 18 tests passing locally and in CI.
- Manual runs with --no-telegram; Telegram send smoke tested.

Notes:
- Config precedence enforced; no hardcoded secrets.
- Payload schema extended without breaking existing keys.

If you want these dropped into files via a ready Cursor prompt, say “cursor doc update prompt” and I’ll paste a single Agent instruction to create/patch all docs. Enjoy the break.

Sources

Commit message — V3.1 Multi-asset foundations (code only)
feat(v3.1): multi-asset foundations — universe scan, stocks, market-hours, digest

- Add multi-asset universe support:
  - config/universe.yaml with crypto + stock symbols
  - scripts/scan_universe.py orchestrator to fan-out runs, rank Top-N, and emit digest
- Symbol + market-hours utilities:
  - symbol_utils.py for normalization and type detection (is_crypto/is_stock/get_symbol_type)
  - trading_hours.py for US equities RTH/EXT/CLOSED and crypto 24x7
- Stock data adapter:
  - bars_stock.py with provider hook, retries, and TB_ALLOW_STUB_BARS=1 fallback
- Digest:
  - digest_utils.py for compact Top-N Telegram/console digest (gap, conf, VolZ, diversity adj, cascade tag, timescale align, sizing)
- Pipeline integration:
  - tracer_bullet: non-breaking additions to set symbol_type and market_hours_state when invoked by orchestrator
- Tests (green):
  - tests/test_universe_loader.py (config loader)
  - tests/test_symbol_utils.py (normalize, type detection; strict UNKNOWN handling)
  - tests/test_trading_hours.py (RTH boundaries, crypto 24x7)
  - tests/test_bars_stock.py (adapter shape/order; zero-lookback guard)
  - tests/test_orchestrator_rank.py (deterministic ranking: |gap| desc, conf desc, symbol asc)
- Fixes/cleanups:
  - Resolved recursion in symbol_utils (normalize_symbol ↔ is_crypto)
  - Deterministic ranking keys and stable sort
  - Trading hours edge cases at session boundaries
  - DASH/UNKNOWN classification made stricter
  - bars_stock zero-division and empty-window safeguards
- CLI examples:
  - TB_NO_TELEGRAM=1 TB_ALLOW_STUB_BARS=1 python3 scripts/scan_universe.py --config config/universe.yaml --top 5 --debug
  - python3 scripts/scan_universe.py --symbols "BTC/USD,AAPL,MSFT" --top 3

Notes:
- Single-symbol scripts/run.py remains unchanged and fully functional
- No documentation updates in this commit (docs will be consolidated later)
- Full test suite passing locally

Sources

