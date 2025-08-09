Here’s a tight, action-focused backlog to pair with tomorrow’s two upgrades.

Core robustness and quality
- Per-source weighting auto-tune
  - Log source-wise precision: for each source, track acceptance rate, FinBERT polarity consistency, and contribution to divergence. Periodically auto-tune weights ±0.02 within [0.9, 1.2].
- Headline polarity coherence check
  - If accepted headlines are highly mixed (e.g., stddev of per-headline sentiment > 0.5), subtract 0.03–0.05 from confidence.
- Duplicate-topic suppression
  - Cluster accepted headlines (mini-batch cosine or simple shingling). If >50% cluster is the same story rewrite, reduce effective count for confidence by 1.

Alpha surface improvements
- Price context guard
  - If price_label is neutral but RSI/MFI extremes detected or key level break (e.g., prior day high/low), add “setup watch” tag and +0.02 to confidence.
- Catalyst tagging
  - Hard-tag catalysts in accepted headlines (ETF flows, SEC actions, halving/mining, hash rate spikes, large on-chain flows). If 2+ headlines share a catalyst, add +0.03 confidence and mention catalyst in alpha_summary.

Trigger and action logic
- Directional filter
  - If FinBERT mean and narrative label disagree in sign (after decay), cut confidence by 0.05. If they agree and both |score| > 0.4, add +0.03.
- Asymmetric trigger
  - Separate triggers for positive vs negative gaps (e.g., bullish trigger 0.95, bearish 1.05) if your backtests show asymmetry.

Data and fetch
- Add CoinTelegraph RSS (optional)
  - Same dedupe, same relevance pipeline; start with weight 1.03 and monitor precision.
- Perplexity prompt variant A/B
  - Maintain two prompts (BTC-only vs BTC+macro) and alternate per run; keep best N items by weighted score.

Persistence and analytics
- JSON schema versioning
  - Add payload_version="v1.1" when you add the new fields (conf_diversity_boost, adaptive_relevance_step, catalyst_tags).
- Metrics log
  - Append a lightweight CSV per run:
    - run_id, accepted_cnt, unique_sources, conf, conf_delta_diversity, threshold, threshold_delta, finbert, decayed_narr, price_score, gap, trigger, action.

Safety and ops
- Fail-quiet guards
  - If any fetch returns 0 but others >0, warn but proceed. Only abort when all sources fail.
- Timeout/backoff config in .env
  - PPLX_TIMEOUT, PPLX_BACKOFF, RSS_TIMEOUT to avoid occasional stalls.

UX/Output
- Alpha summary enrichment
  - Include top catalyst (if any) and “what would change the action” (e.g., “gap > trigger or volume_z > 0.7”).
- Alert hooks (optional)
  - If abs(gap) within 0.1 of trigger and conf > 0.6, emit an alert line to stdout and write runs/<id>_alert.txt.

Backtest hooks (prep)
- Snapshot inputs per run
  - Save accepted headlines with timestamps and per-headline FinBERT scores. This enables later PnL simulation on divergence crossings.

If you want, I can provide full scripts for:
- Confidence boost by source diversity and polarity coherence
- Adaptive threshold controller with state (reads last run’s accepted count and adjusts RELEVANCE_THRESHOLD)
- Catalyst tagging and inclusion in alpha_summary
- CoinTelegraph RSS fetcher and integration
- Schema versioning and metrics CSV writer

Sources
