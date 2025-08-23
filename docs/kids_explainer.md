# Confidence Engine for Kids — The Friendly Market Detective

> A simple, fun guide to how our agent watches crypto coins and tells helpful stories about them.

---

## 1) Meet Confidence Engine
- Confidence Engine is like a weather helper for coins.
- It reads the news (clouds) and checks prices (thermometer).
- When the story says “sunny!” but the thermometer says “cold,” it thinks, “Hmm, something interesting might happen.”

---

## 2) What does it actually do?
- Reads trusted news about coins like Bitcoin or Ethereum.
- Looks at price movement across different times (like the last hour, day, and week).
- Tries to spot a “disagreement” between the news and the price.
- Makes a Top List of coins where the disagreement is strongest.
- Writes a short reason (“evidence line”) and a simple plan (up, down, or sideways) for each coin.
- Saves everything neatly so we can check later. It can also send a friendly message to Telegram or Discord (if we turn that on).

---

## 3) How does Confidence Engine check itself?
- It looks back later and asks: “Was I right?”
- It checks 1 hour, 4 hours, and 1 day later.
- It keeps score in a notebook (hit‑rate) so we can see if it’s learning.
- If something goes wrong (like missing price data), it writes a little note about why.
- Every night, it updates its notebook automatically.

---

## 4) Why is a disagreement useful?
- People sometimes overreact to news or ignore it.
- If the news is positive but the price hasn’t moved yet, maybe the price will catch up.
- If the news is negative but the price is still high, maybe the price could fall.
- These little mismatches can turn into opportunities.

---

## 5) How it stays safe and fair
- Number‑free chat: messages avoid hard numbers, so it’s easy to read and safe to share.
- Full details in files: saved files keep the precise info so we can study carefully.
- Safe mode: we can run it quietly with no chat messages and without pushing anything to the internet.
- Reproducible test: it can check that two runs match when nothing changed (consistency gate).

---

## 6) The detective’s toolkit (kid view)
- News glasses: reads structured news summaries.
- Price compass: checks if price is going up, down, or flat on different time windows (1h, 4h, 1D, 1W).
- Honesty checkers: makes sure the news story and the price story agree (or notices when they don’t).
- Evidence maker: writes a short, human‑friendly reason you can understand.
- Memory book: saves a neat file with the story, plan, and results so we can learn over time.

---

## 7) What it saves for us
- A file for each big run with:
  - Evidence line: the one‑sentence reason.
  - Thesis: what we think (action), how risky it feels (risk band), and how ready we are (readiness).
  - Plan by timeframe: for 1h/4h/1D/1W it keeps entries, invalidation (what proves us wrong), and targets, plus a short "Outcome" explanation.
  - Optional Polymarket notes: what prediction markets say (numbers stay in files, not in chat).

These files live in the `universe_runs/` folder next to your project.

---

## 8) A tiny example story
- The news says: “Builders are launching something big on Ethereum this week.”
- The price is still flat today.
- Confidence Engine says: “News is sunny but the thermometer is cool. ETH might warm up soon.”
- It writes a short plan and saves it. If chat is on, it sends a friendly message.

---

## 9) How to use it safely (with a grown‑up)
- Make sure chat is off if you don’t want messages:
```
TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 \
python3 scripts/tracer_bullet_universe.py --config config/universe.yaml --top 10
```
- To double‑check honesty, run the “consistency gate” (it should match twice when nothing changes):
```
TB_DETERMINISTIC=1 TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 \
TB_UNIVERSE_GIT_AUTOCOMMIT=0 TB_UNIVERSE_GIT_PUSH=0 \
python3 scripts/consistency_check.py --config config/universe.yaml --top 10
```

---

## 10) Words you might hear (Kid Glossary)
- Narrative: the story news is telling.
- Price action: how the coin’s price moves.
- Divergence: when the story and the price don’t match.
- Evidence line: one friendly sentence that explains the idea.
- Timeframe (TF): a window of time, like 1 hour (1h) or 1 day (1D).
- Invalidation: what would prove the idea wrong, so we can stop and rethink.

---

## 11) What the grown‑ups like about it
- It’s explainable: always gives a simple reason.
- It’s careful: keeps numbers in files but keeps chat friendly.
- It learns: by saving files, we can study what worked and improve.
- It’s reliable: has a test to be consistent and safe modes to avoid surprises.

---

## 12) Quick Q&A
- Q: Does it buy coins by itself?
  - A: Not by default. It mostly explains and saves. Grown‑ups can choose what to do next.
- Q: Can it be wrong?
  - A: Yes. That’s why it writes an invalidation so we can change our mind safely.
- Q: Can I see the details?
  - A: Yes. The saved files in `universe_runs/` have the full story for study time.

---

## 13) Last thought
Think of Confidence Engine like a kind, honest friend who reads a lot, watches calmly, and writes down clear, simple reasons. It helps you learn and make better choices over time.
