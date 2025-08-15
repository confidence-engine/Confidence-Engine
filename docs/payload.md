# Payload reference

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
```
import json
# Replace with a real file path
# d = json.load(open("runs/<id>.json"))
# print(d["alpha_summary"], d["confidence"], d.get("source_diversity"))
```

## V3 additions

### timescale_scores
- Structure:
```
{
  "short": {"divergence": float, "price_change_pct": float, "volume_z": float},
  "mid": {"divergence": float, "price_change_pct": float, "volume_z": float},
  "long": {"divergence": float, "price_change_pct": float, "volume_z": float},
  "combined_divergence": float,
  "aligned_horizons": int,
  "alignment_flag": bool,
  "weights": {"short": float, "mid": float, "long": float}
}
```
- Notes:
  - divergence_h uses the current decayed narrative as proxy for each horizon
  - price_score_h is tanh(3.0 * signed_pct), signed_pct=(last-first)/first*100

### confirmation_checks and confirmation_penalty
- confirmation_checks: list of {name: str, passed: bool, delta: float}
- confirmation_penalty: float (≤ 0), clamped to TB_CONF_PENALTY_MIN ≤ x ≤ 0
- Checks implemented:
  - price_vs_narrative
  - volume_support
  - timescale_alignment

### position_sizing
- Structure:
```
{
  "confidence": float,
  "target_R": float,
  "notes": [str, ...],
  "params": {
    "conf_floor": float,
    "conf_cap": float,
    "min_R": float,
    "max_R": float,
    "used_vol_norm": float | null
  }
}
```
- Behavior:
  - target_R=0 when confidence < conf_floor
  - Linear interpolation from [conf_floor, conf_cap] to [min_R, max_R]
  - Optional vol normalization (disabled by default)

## V3.1 additions

### symbol_type and market_hours_state
- symbol_type: "crypto" | "stock" | "unknown"
- market_hours_state: "RTH" | "CLOSED" | "24x7" | "UNKNOWN"
- Added to payload when available from symbol analysis

## V3.2 additions

### evidence_line
- Per-payload human evidence summary line attached post-scan. May be `null` if disabled/absent.

### thesis snapshot (persisted subset)
- Minimal stable fields persisted into each payload for chat/UI parity:
  - `thesis.action`: "Buy" | "Sell" | "Watch"
  - `thesis.risk_band`: "Low" | "Medium" | "High"
  - `thesis.readiness`: "Now" | "Near" | "Later"

### per-timeframe plan schema (TF plan)
- When analysis levels are available (preferred) or synthesized/fallback, each payload may include a `plan` with keyed timeframes:
```
plan: {
  "1h": {
    "entries": [
      {"type": "trigger" | "fade", "zone_or_trigger": float | [float, float]}
    ],
    "invalidation": {"price": float, "condition": "close below" | "close above" | "breach"},
    "targets": [{"label": "TP1", "price": float}, ...],
    "source": "analysis" | "fallback",
    "explain": str,
    "context": {"structure_hint": str | null, "pattern_hint": str | null, "weekly": "demand" | "supply" | null}
  },
  "4h": {...},
  "1D": {...},
  "1W": {...}
}
```
- Notes:
  - `analysis` plans are derived from agent signals/levels; `fallback` plans are heuristic using spot price.
  - Crypto payloads may include full TF plans; stocks typically omit auto plans and only persist when explicitly provided.

### Universe file enrichments
- The saved universe artifact (see `universe_runs/*.json`) is enriched post-scan with:
  - Per-payload `evidence_line` and persisted `thesis` subset (see above)
  - Optional `plan` snapshot (per TF) when available/derived
  - Top-level `polymarket`: array of simplified market items:
```
polymarket: [
  {
    "market_name": str,
    "stance": str | null,
    "readiness": str | null,
    "edge_label": str | null,
    "rationale": str | null,
    "implied_prob": float | null,
    "implied_pct": float | null,
    "tb_internal_prob": float | null,
    "liquidity_usd": float | null,
    "event_end_date": str | null,
    "market_id": str | null,
    "quality_score": float | null
  }, ...
]
```

### Consistency and persistence notes
- Consistency gate: CI runs a deterministic double-scan comparer to ensure identical payloads/rankings (timestamps ignored).
- Persistence: universe artifacts and `universe_runs/metrics.csv` are auto-committed and optionally pushed, controlled by env:
  - `TB_UNIVERSE_GIT_AUTOCOMMIT` (default "1")
  - `TB_UNIVERSE_GIT_PUSH` (default "1")
  - `TB_UNIVERSE_WRITE_METRICS` (default "1")


