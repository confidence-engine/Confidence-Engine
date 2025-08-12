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

V3.1 universe mirroring + git integration feat(v3.1): universe scan mirroring and opt-in git auto-commit/push
	•	scripts/scan_universe.py:
	•	Add TB_UNIVERSE_MIRROR_TO_RUNS=1 to copy universe_runs file into runs/
	•	Add TB_UNIVERSE_GIT_AUTOCOMMIT=1 to git add/commit universe file(s)
	•	Add TB_UNIVERSE_GIT_PUSH=1 to push after commit (requires AUTOCOMMIT)
	•	Safe defaults (all off), robust try/except around git ops, never abort scan
	•	Consistent “Universe” logs; import cleanup
	•	Verified:
	•	Mirror-only, auto-commit, and auto-commit+push flows
	•	Default flow unchanged (no mirror/commit)
	•	Full test suite remains green

Dev log update
	•	Added number-free, crypto-first human digest
	•	Created scripts/digest_formatter.py to render a conversational digest with BTC/ETH first, entries/exits, levels-to-watch, and risk-based sizing.
	•	Maps internal metrics to qualitative bands (polarity, confidence, volume state, alignment, quality tags, readiness, sizing) without exposing numbers.
	•	Includes Playbook footer and equities-as-background section.
	•	Integrated formatter into single-run flow
	•	Edited scripts/scan_universe.py to call render_digest(summary) after writing universe artifacts.
	•	Prioritizes sending the human digest to Telegram; also prints to console.
	•	Honors existing artifacts write; no analyzer logic changes.
	•	Optional prompt/style reference
	•	Added scripts/prompts/digest_prompt.md documenting template, tone, and rules for the digest.
	•	CLI/env control
	•	Added optional runtime toggle (–no-human-digest) and environment variable support (TB_HUMAN_DIGEST) to enable/disable human digest without code changes.
	•	Verification
	•	Test run confirmed human digest generation and Telegram delivery; artifacts still written to universe_runs/.


  ------------------------------------------------
  # Tracer Bullet — Comprehensive Roadmap Update (what’s done, what’s planned, and alignment)

Below is a consolidated, milestone-based roadmap compiled from our conversation history, organized by version, with explicit completion status, what shipped in each version, what’s next, and a clear alignment verdict versus the initial vision.

## Executive verdict

- Alignment: The project remains aligned with the original “Tracer Bullet” approach and the objective to build an interpretable, evidence-driven alpha engine that blends narrative with price/volume and ships incrementally with auditability and guardrails. We are not building “something else”; we’ve deepened exactly what we set out to do: reliability first, explainability, multi-asset foundations, and human-readable outputs.  
- Scope adjustments: Two timeline corrections were made for clarity, not direction changes:  
  - 24/7 cloud scheduling is now explicitly a later milestone (v6) after full testing, rather than earlier.  
  - v3.3 expands to all available crypto alts (not a small subset).  

## Completed milestones

### v1 — Hardening & Reliability [DONE]
- Preflight/health checks to validate environment and Telegram reachability; automatic directory setup for artifacts.  
- CLI wrapper with clear precedence (CLI > env > .env > defaults), centralized structured logging, INFO/DEBUG modes.  
- Robust Telegram delivery: plain-text default, truncation safeguards, graceful handling of 200/400/429, opt-out via flag.  
- Artifact retention: pruning runs/ and bars/ by most-recent N files; configurable via env.  
- Tests and CI: unit tests for divergence, payload schema, Telegram formatting, directory checks; CI lint/test flow.  
- Documentation: README, RUNBOOK, CONTRIBUTING, .env.example, payload docs.  

Why this matters: Establishes a dependable, reproducible core loop with safe ops, visibility, and clean artifacts.

### v2 — Crowd Immunity [DONE]
- Source Diversity Engine: shapes confidence by unique sources and penalizes “echo chambers”; artifacts added to payload.  
- Cascade/HYPE Detector: flags repetitive narrative lacking quant confirmation; bounded confidence deltas; payload enriched.  
- Contrarian Viewport: informational tag for potential crowd mistakes under certain narrative/price conditions; included in payload and one-liners.  
- Tests for diversity/cascade/contrarian presence and behavior.

Why this matters: Reduces herd-driven noise; improves robustness and interpretability of narrative signals.

### v3 — Bias Immunity + Sizing [DONE]
- Multi-timescale scoring: short/mid/long metrics and combined view with alignment gating.  
- Negative-confirmation checks: structured penalty logic with clamps; transparent reasons in payload.  
- Informational position sizing: confidence-to-R mapping, optional vol-normalization; reported without forcing trades.  
- Telegram lines reflect timescales, penalties, and sizing guidance; tests cover blending, clamps, and boundaries.

Why this matters: Adds disciplined structure to confidence, avoids over-trust in contradictory evidence, and connects confidence to position logic.

### v3.1 — Multi-Asset Foundations (Crypto + Stocks) [DONE]
- Universe configuration for symbols; symbol utilities for normalization and type detection.  
- Trading-hours awareness (crypto 24/7 vs equities RTH/extended/closed).  
- Stock bars adapter scaffold (with safe fallbacks), orchestration for multi-symbol scan, top-N ranking, digest utilities.  
- Payload extensions: symbol_type, market_hours_state, timescale presence.  
- Universe runs written to universe_runs/ with timestamping; optional mirroring to runs/.  
- Git integration hooks implemented behind env gates (off by default) for mirror/commit/push; robust try/except and logging.  
- Tests: universe loader, symbol utils, trading hours, stock adapter shape/safety, ranking determinism.

Why this matters: Enables consistent multi-asset scanning and reporting without breaking single-asset flow.

### v3.1.x — Human Digest Integration (crypto-first, number-free) [DONE]
- Added number-free, conversational digest formatter producing a consistent crypto-first report (BTC/ETH prioritized), including levels-to-watch (descriptive), entries/exits, and risk-based sizing bands.  
- Integrated into single-run flow: produced after artifacts are written, sent to Telegram, and printed to console; analyzer logic remains unchanged.  
- Optional prompt/style reference file; optional toggle to enable/disable digest output; safe defaults preserved.  

Why this matters: Delivers a human-ready narrative output without exposing raw metrics, boosting usability for decision-making while keeping the quantitative engine intact.

## In progress

### v3.1.x — Auto-commit/push hardening [IN PROGRESS]
- Goal: Ensure universe_runs/*.json and universe_runs/metrics.csv are staged, committed, and pushed automatically when env gates are on.  
- Current status: Commit/push plumbing exists behind env flags, but defaults are OFF; some terminals may miss staging for metrics.csv; logs need explicit “Auto-commit done.” / “Pushed.” confirmations.  
- Next steps: Confirm staging includes both JSON and metrics; add explicit result logs; verify non-interactive push across environments.

Why this matters: Eliminates manual staging/pushing and keeps repo artifacts consistent across runs and environments.

## Planned milestones

### v3.2 — Reliability Hardening (agent, non-24/7)
- Retries/backoff for transient providers; structured error handling and graceful skips.  
- Schema checks and digest self-check for thin or missing inputs; produce useful outputs even when evidence is sparse.  
- Clear alert notes when runs are skipped or degraded.

Why this matters: Improves run resilience and developer/operator trust before moving to continuous scheduling.

### v3.3 — Full Crypto Alt Coverage + Evidence Lines
- Expand coverage to all available liquid crypto alts (not just a few), using the same number-free template.  
- Add brief “why now” evidence lines per BTC/ETH and key alts, describing sentiment/news/structure vs price in plain English (no numbers, no links).  
- Maintain crypto-first priority and keep equities de-emphasized.

Why this matters: Completes crypto breadth while preserving interpretability, providing rationale for attention and bias.

### v3.4 — Execution Quality (paper/dry-run)
- Microstructure-aware tactics (market vs limit vs slices by spread/volatility) and cool-downs to avoid clustering.  
- Volatility-aware sizing with conservative caps.  
- Measure slippage versus baseline to confirm improvements.

Why this matters: Turns good signals into better realized outcomes while staying in a safe, non-live mode.

### v4 — Backtesting & Governance
- Event-ordered replay for bars+headlines; walk-forward validation; cohort analytics (asset/time-of-day/volatility regime).  
- Parameter governance cadence with documented thresholds from out-of-sample.  
- Reproducible backtests with clear in/out-of-sample splits.

Why this matters: Converts plausible intuition into evidence-backed settings and reduces hidden look-ahead risk.

### v4.1 — Paper Execution & Risk Controls
- Paper order lifecycle with audit logs; portfolio caps; kill-switches and guardrails.  
- Idempotency, reproducibility tags per decision/version.

Why this matters: Operational discipline before any live risk, ensuring safe failure modes.

### v5 — Data Breadth & Explainability+
- Optional attention/crowd proxies as secondary evidence (controlled, never primary).  
- Source credibility learning; compact case files per signal for audits/postmortems.

Why this matters: Improves precision and review speed without sacrificing interpretability.

### v6 — 24/7 Cloud Agent Run (after full testing)
- GitHub Actions (or equivalent) scheduled workflows: crypto-only every 15 minutes; mixed hourly with staggered minute.  
- Secrets management; non-interactive push; deterministic cadence; Telegram delivery.  
- Monitoring/rollback for the scheduler jobs.

Why this matters: Moves to truly autonomous operation only after we’ve finished hardening, coverage, and testing.

### v7 — Live Capital (small, guarded)
- Strict loss limits, circuit breakers, anomaly alerts, version tagging; limited deployment scope.  
- Rollback rehearsed; postmortem-ready artifacts.

Why this matters: Begin live exposure safely, learning from real frictions without over-scaling.

## Are we on plan?

- Yes, with a clarified timeline: Up through v3.1.x we are on track and consistent with the tracer-bullet philosophy—thin end-to-end, then harden, then expand coverage, then automate scheduling, then consider live.  
- The only course correction was to explicitly place 24/7 scheduling at v6 after testing, and to broaden v3.3 to cover all available alts; both are alignment fixes, not directional changes.

## Operational notes (scheduling and automation guardrails)

- When we reach v6, scheduled workflows can use cron-based triggers with sensible intervals; GitHub Actions supports 5-minute minimum cadence and may delay around top-of-hour loads, so staggered minutes are recommended to reduce contention[1][2][3].  
- If we explore interim in-app scheduling for dev or server use, APScheduler’s cron/interval triggers and background schedulers are a robust option before moving to managed schedules[4][5][6][7][8].  

## What to do next (immediate focus)

- Finish v3.1.x hardening: confirm staging includes metrics.csv and JSON; add explicit commit/push logs; test non-interactive push.  
- Start v3.2: implement retries/backoff and schema/digest self-checks; ensure graceful degradation and actionable logs when inputs are thin.  
- Prepare v3.3 backlog for “all alts + evidence lines” with the digest template unchanged in tone and structure.

These steps preserve our reliability-first approach and set us up for a smooth v6 shift to 24/7 automation after full testing.

Sources
[1] Why does my cron configured GitHub Action not run every 2 minutes? https://stackoverflow.com/questions/63192132/why-does-my-cron-configured-github-action-not-run-every-2-minutes
[2] How to Schedule Workflows in GitHub Actions - DEV Community https://dev.to/cicube/how-to-schedule-workflows-in-github-actions-1neb
[3] Run your GitHub Actions workflow on a schedule - Jason Etcovitch https://jasonet.co/posts/scheduled-actions/
[4] User guide — APScheduler 3.11.0.post1 documentation https://apscheduler.readthedocs.io/en/3.x/userguide.html
[5] Job Scheduling in Python with APScheduler | Better Stack Community https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/
[6] Scheduled Jobs with Custom Clock Processes in Python with ... https://devcenter.heroku.com/articles/clock-processes-python
[7] Python Job Scheduling: Methods and Overview in 2025 https://research.aimultiple.com/python-job-scheduling/
[8] I Replaced Cron Jobs with Python Schedulers | by Muhammad Umar https://python.plainenglish.io/i-replaced-cron-jobs-with-python-schedulers-6a25f94bd642
[9] Tracer Bullets - C2 wiki https://wiki.c2.com/?TracerBullets
[10] How Tracer Bullets Speed Up Software Development | Built In https://builtin.com/software-engineering-perspectives/what-are-tracer-bullets
[11] Tracer-Bullet — Why we should build features during discovery https://thedigitalbusinessanalyst.co.uk/tracer-bullet-why-we-must-build-features-during-discover-952df9c5a65b
[12] bullet-scraper/scrapes/afbulletsafe.txt at master - GitHub https://github.com/AF-VCD/bullet-scraper/blob/master/scrapes/afbulletsafe.txt
[13] Caitlin Hudon - Tracer bullets + working backwards - YouTube https://www.youtube.com/watch?v=vNZY0zhg3Do
[14] How do you make a workflow run randomly within a given time period? https://github.com/orgs/community/discussions/131450
[15] [PDF] Go: Building Web Applications - anarcho-copy https://edu.anarcho-copy.org/Programming%20Languages/Go/Go%20building%20web%20application.pdf
[16] [PDF] EXPRESSION OF INTEREST - BECIL https://www.becil.com/uploads/topics/17193916113963.pdf
[17] GitHub Actions Cron Schedule for Running Once in 2 Weeks #158356 https://github.com/orgs/community/discussions/158356
[18] The Evolving Landscape of Antibody–Drug Conjugates: In Depth ... https://pubs.acs.org/doi/10.1021/acs.bioconjchem.3c00374
[19] [PDF] DOT&E FY2021 Annual Report https://www.dote.osd.mil/Portals/97/pub/reports/FY2021/other/2021DOTEAnnualReport.pdf
[20] Apscheduler is skipping my task. How to eliminate this? https://stackoverflow.com/questions/73343854/apscheduler-is-skipping-my-task-how-to-eliminate-this

Here you go.

Commit message (conventional, concise)
feat(digest): add Weekly + Engine Telegram digest with crypto-only prices; stocks headers-only

- env: add TB_HUMAN_DIGEST, TB_NO_TELEGRAM, TB_DIGEST_INCLUDE_WEEKLY, TB_DIGEST_INCLUDE_ENGINE, TB_DIGEST_MAX_TFS, TB_DIGEST_DRIFT_WARN_PCT, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
- formatter: new scripts/tg_digest_formatter.py (Weekly/Engine sections; crypto shows spot/entries/targets; stocks suppress prices/levels)
- weekly/engine: new scripts/tg_weekly_engine.py (build_weekly_overview, build_engine_minute)
- telegram: new scripts/tg_sender.py (gated send with retries, respects TB_NO_TELEGRAM)
- runner: wire scripts/tracer_bullet_universe.py to assemble assets_data/order, render digest, optional send
- scan: return payloads for downstream digest formatting
- docs: .env.example updated

dev logs update
Summary
- Implemented a new plain-text Telegram digest including Weekly Overview and Engine in One Minute.
- Crypto assets (BTC/ETH first) show spot narrative and plans (entries/targets/invalidation). Stocks show headers/notes only—no prices/levels.
- Telegram sending added with env gating and simple retry/backoff.

Details
- Env
  - Added flags to .env.example:
    - TB_HUMAN_DIGEST (enable human digest behavior)
    - TB_NO_TELEGRAM (gate sending; default skip)
    - TB_DIGEST_INCLUDE_WEEKLY / TB_DIGEST_INCLUDE_ENGINE
    - TB_DIGEST_MAX_TFS (default 2)
    - TB_DIGEST_DRIFT_WARN_PCT (default 0.5)
    - TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
- New modules
  - scripts/tg_weekly_engine.py
    - build_weekly_overview(): derives regime, up to two anchors, plan_text, optional catalysts from existing signals
    - build_engine_minute(): thesis/evidence/action and compact narrative stats (no numbers/probabilities)
  - scripts/tg_digest_formatter.py
    - is_crypto() helper
    - render_digest(): Title, Executive Take, Weekly, Engine, per-asset blocks (crypto full; stocks headers-only), Playbook
    - Drift warning applied when abs(drift) > TB_DIGEST_DRIFT_WARN_PCT
  - scripts/tg_sender.py
    - send_telegram_text(): POST with retries (0.5s/1s/2s), respects TB_NO_TELEGRAM and 429 Retry-After
- Wiring
  - scripts/tracer_bullet_universe.py updated to:
    - Build assets_data and assets_ordered (BTC, ETH, other crypto, then stocks)
    - Generate weekly/engine
    - Render digest and print
    - If TB_HUMAN_DIGEST=1 and TB_NO_TELEGRAM=0, send via Telegram
  - scripts/scan_universe.py returns payloads for downstream formatter consumption; internal Telegram disabled to avoid double-send
- Behavior notes
  - No refactor of analyzers; formatter degrades gracefully on missing fields
  - No numeric model metrics/probabilities printed in digest
  - Equities remain in universe; they render without prices/levels by design
- Quick test (print-only)
  - TB_HUMAN_DIGEST=1 TB_NO_TELEGRAM=1 python3 scripts/tracer_bullet_universe.py
  - Observed: Weekly + Engine sections; BTC/ETH with spot and plans; SPY/AAPL/MSFT headers/notes only; send skipped as expected
- How to send
  - export TB_HUMAN_DIGEST=1
  - export TB_NO_TELEGRAM=0
  - export TELEGRAM_BOT_TOKEN=...
  - export TELEGRAM_CHAT_ID=...
  - python3 scripts/tracer_bullet_universe.py

Follow-ups
- Optional tests: ensure stocks never print numeric lines; ensure crypto blocks include entries/targets when present.
- If you don’t want run artifacts tracked, add universe_runs/ to .gitignore.

Sources

Commit message (conventional)
feat(digest): crypto numeric prices in Telegram digest; keep stocks narrative-only

- formatter: show Spot/Entries/Invalidation/Targets for crypto; suppress numeric levels for stocks
-  wire crypto spot/levels into assets_data; format per-tf numeric entries (price or L–H), invalidation (price+condition), targets (TPn)
- sender: gated Telegram send with retries and backoff; respects TB_NO_TELEGRAM
- ordering: BTC, ETH, other crypto, then stocks
- env: enable TB_DIGEST_INCLUDE_PRICES; keep weekly/engine toggles (weekly not yet implemented)

Developer log (detailed)
Summary
- Implemented numeric pricing for crypto assets in the Telegram Human Digest. Stocks remain header/notes only with no numeric levels.
- Digest renders Spot price, per-timeframe Entries/Invalidation/Targets for BTC/ETH using current provider integration. Weekly overview is not implemented yet (placeholder/stub still prints regime/plan text if present); Engine in One Minute section remains narrative.

What changed
- tg_digest_formatter.py
  - is_crypto() used to branch rendering: crypto prints numeric Spot and plan levels; stocks print only structure/sizing narrative.
  - Crypto per-timeframe:
    - Entries: supports numeric trigger price or L–H zone.
    - Invalidation: prints numeric price with condition suffix (e.g., “1h close below”).
    - Targets: prints TP labels with numeric prices.
  - Spot line: prints numeric price for crypto; optional drift warning if threshold exceeded.
  - Stocks path: header + Structure + Sizing only; no Spot/Entries/Targets/Weekly/Drift lines.

- tracer_bullet_universe.py
  - Assembles assets_data with numeric fields for crypto:
    - spot: float
    - plan[tf].entries: trigger price or zone [low, high]
    - plan[tf].invalidation.price: float + condition string
    - plan[tf].targets[].price: floats
  - Maintains ordering: BTC, ETH, other crypto, then stocks.
  - Telegram send call remains gated by TB_HUMAN_DIGEST and TB_NO_TELEGRAM.

- tg_sender.py
  - send_telegram_text(text): skips when TB_NO_TELEGRAM=1 or creds missing; retries 3x with incremental backoff; honors Retry-After on 429.

Environment
- Ensure these are present in .env/.env.example (weekly not yet wired end-to-end):
  - TB_HUMAN_DIGEST=1
  - TB_NO_TELEGRAM=1 (set 0 to send)
  - TB_DIGEST_INCLUDE_PRICES=1
  - TB_DIGEST_INCLUDE_WEEKLY=1 (weekly section pending full implementation)
  - TB_DIGEST_INCLUDE_ENGINE=1
  - TB_DIGEST_MAX_TFS=2
  - TB_DIGEST_DRIFT_WARN_PCT=0.5
  - TELEGRAM_BOT_TOKEN=
  - TELEGRAM_CHAT_ID=

Current behavior (confirmed by latest run)
- Crypto:
  - BTC/USD shows Spot, numeric Entries, numeric Invalidation (with side/condition), numeric Targets.
  - ETH/USD shows Spot, numeric L–H entry zone, numeric Invalidation, numeric Targets.
- Stocks: SPY/MSFT/AAPL show only header + structure/sizing narrative; no numeric prices or levels.
- Executive Take and Engine sections render; Weekly shown only if data provided (full logic pending).
- Telegram digest text is generated; sending controlled by env.

Known gaps / next steps
- Weekly Overview: implement actual weekly regime/anchors extraction; currently minimal/placeholder. Wire build_weekly_overview to real signals and pass anchors (supply/demand [low, high]) when available.
- Numeric rounding: consider compact formatting (e.g., 118,877.07 → 118.88K) via formatter option.
- Drift guard: ensure drift computation uses snapshot vs. current spot consistently; expose threshold from env.

How to test
- Local print only:
  - TB_HUMAN_DIGEST=1 TB_NO_TELEGRAM=1 TB_DIGEST_INCLUDE_PRICES=1 python3 scripts/tracer_bullet_universe.py
  - Expect BTC/ETH numeric Spot/Entries/Invalidation/Targets; stocks without numbers.
- Send to Telegram:
  - TB_HUMAN_DIGEST=1 TB_NO_TELEGRAM=0 TB_DIGEST_INCLUDE_PRICES=1 TELEGRAM_BOT_TOKEN=xxx TELEGRAM_CHAT_ID=yyy python3 scripts/tracer_bullet_universe.py

Roll-back plan
- Toggle TB_DIGEST_INCLUDE_PRICES=0 to suppress numeric printing while keeping digest structure intact.
- Revert to previous digest by disabling the new formatter path in tracer_bullet_universe.py (guard by env flag if needed).

Sources
