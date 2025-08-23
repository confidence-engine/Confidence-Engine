# I built a calm AI that reads crypto’s mood. Here’s how it works—and a free tool for you.

Most retail trades the headline they just saw, chases green candles, and anchors to the loudest number in the room. I spent 5+ years watching the same pattern repeat: the story says “go,” the tape says “meh,” and the gap between emotion and math is where real edge hides. I built an agent to live in that gap—measured, explainable, and crowd‑aware.

This is the craft behind it. The agent is open, free, and intentionally a bit rough because it’s experimental by design.

## What Confidence Engine is
- A calm research agent that compares narrative (news) with tape (price/volume).
- Flags meaningful divergences and writes a human‑readable plan with explicit invalidation.
- Current emphasis: BTC and ETH first; expands only after reliability and data quality prove out.
- Saves every decision and reason to files so you can audit, learn, and improve.
- Not limited to crypto: the methodology generalizes to traditional markets (equities, ETFs, sectors, FX, rates). Polymarket integration is optional.

## Why typical retail gets trapped
- Chases the latest headline and overreacts to single sources.
- Anchors to numbers in chat (percentages, price targets) and stops thinking critically.
- Trades “sentiment” without checking structure (volume, timescale alignment).
- Has no explicit invalidation, so bad positions linger.

## How the agent avoids those traps (wisdom‑of‑crowds, without the herd)
- Diverse sources, not loudest voices: fuses multiple reputable narratives; collapses echo chains to prevent “one‑article truth.”
- Prediction markets as reference, not oracle: Polymarket is read‑only context, hidden in chat by default to avoid anchoring.
- Structure first, hype last: timescale alignment and volume support must confirm narrative before conviction rises.
- Bounded confidence and explicit invalidation: every stance carries “what proves us wrong.”

## How it achieves this
- Reads fresh news and filters per‑asset for true relevance.
- Scores narrative momentum vs multi‑timescale price/volume behavior.
- Applies confirmation gates (timescale alignment, volume support, narrative–price consistency).
- Shapes confidence conservatively; chat stays number‑free to reduce anchoring. Full numeric detail lives in saved artifacts.

## Sample digest (number‑free chat, BTC/ETH focus)
```
Executive Take: Mixed day. Leaders skew long; wait for clean triggers.

BTC — Watch
Evidence: L2 infra headlines build momentum; tape remains range‑bound intraday.
Plan: 1H reclaim over prior VAH; invalidate on loss of mid. Patience until volume confirms.
Notes: Divergence present but short-term structure not aligned.

ETH — Watch →
Evidence: Ecosystem shipping catalysts; 1D momentum outpacing spot drift.
Plan: Prefer 4H breakout retest; invalidate on failed retest. Respect invalidation first.
Notes: Polymarket read‑only: market in‑line; stand aside until structure confirms.
```

Artifacts in `universe_runs/` contain evidence lines, invalidation, targets, provenance, and numeric fields for evaluation.

## AI tech stack and engineering — LLM + finance NLP, built for speed and signal
- LLM narrative synthesis (Perplexity API)
  - Purpose‑built prompts generate concise, source‑tagged evidence lines per asset.
  - Bias‑aware phrasing avoids false precision and pushes explicit invalidation.
  - Cross‑source consensus fuses multiple reputable sources; collapses near‑duplicates to reduce echo risk.
- Finance‑tuned NLP (FinBERT + utilities)
  - Finance‑aware cues complement LLM summaries; sentiment is treated as a prior, not a trade trigger.
- Divergence engine (narrative vs tape)
  - Multi‑timescale features quantify structure; divergence measures story–price mismatch.
  - Confirmation gates enforce discipline: price‑vs‑narrative sanity, volume support, timeframe alignment.
- Confidence shaping with guardrails
  - Confidence is bounded and explainable with concrete penalty reasons (e.g., “volume not confirming,” “timescales misaligned”).
  - Source weighting and crowd‑immunity heuristics down‑rank hype/cascade bursts.
- Fast, “snappy” operation
  - Heavy reasoning (LLM, prediction‑market text) offloaded to APIs; local logic stays lean for scoring, gating, formatting.
  - Relevance filtering and a curated universe keep latency low and outputs readable.
- Provenance and evaluation
  - Every run emits structured artifacts with provenance for audit and reproducibility.
  - Calibration (Brier/log‑loss) on resolved datasets enables cohort studies and steady improvement.

## Tech stack used to build
- Perplexity API for LLM‑based narrative synthesis
- Transformers + Torch for FinBERT and classifier/embedding primitives
- spaCy + sentence-transformers for relevance and semantic de‑duplication
- pandas + numpy for multi‑timescale feature engineering and divergence math
- Alpaca Trade API for market data and (optional) execution plumbing in live mode
- requests + httpx for robust API integrations
- python‑dotenv + PyYAML + pydantic for clean config and schema validation
- python‑telegram‑bot (and Discord equivalent if enabled) for clean, number‑free digests
- APScheduler for safe, staggered scheduling when reliability is proven
- matplotlib for quick evaluation visuals; pytest for sanity tests

## How to use it (quick start)
- Requirements: Python 3, Perplexity API key. Telegram/Discord optional.
- Setup:
  1) Clone the repo; copy `.env.example` to `.env`.
  2) Add your keys; start in safe mode to avoid sends.

- Safe run (no messages, no trades):
```bash
TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 \
python3 scripts/tracer_bullet_universe.py --config config/universe.yaml --top 10
```

- You’ll get:
  - Saved artifacts in `universe_runs/` (evidence, invalidation, targets, provenance).
  - Optional digests to Telegram/Discord when you later enable sends.

## Live agent and performance tracking
- Live operations: The agent can run with safeguards and produce real‑time digests; actions are gated by readiness rules.
- Transparent by default: Runs emit structured artifacts you can audit (`universe_runs/`, `runs/`, `eval_runs/`).
- Road ahead: Rigorous testing and periodic summaries so results remain transparent and falsifiable.

## Not limited to crypto
- The methodology generalizes beyond BTC/ETH and Polymarket. The story–price divergence framework applies to:
  - Equities (single‑names and sectors)
  - ETFs and indices
  - FX and rates
  - Commodities with narrative pipelines

## Not financial advice—use at your own risk
- This is a research/learning tool. Markets are volatile.
- You are responsible for your decisions. Start in safe mode. Read artifacts. Build conviction.

## Important experimental disclaimer
- Results may vary and will not be perfectly consistent during this experimental/testing phase. Expect iteration and guardrail tuning over the coming months.

## Call to action
- Docs: see `README.md` and `README_CLEAN.md`
- Start in safe mode. Study artifacts. Track results. Iterate responsibly.
