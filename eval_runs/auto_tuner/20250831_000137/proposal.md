# Auto-tuner Proposal (propose-only)

## Candidate config
- stop_mode: atr_fixed
- atr_mult: 2.0
- atr_period: 10
- tp_pct: 0.03
- sl_pct: 0.01
- time_cap_bars: 0
- risk_frac: 0.01

## Metrics
- Sharpe: -0.2813554574622013
- CAGR: -0.0017226917855827084
- Max DD: 8.665783901514797e-05
- Trades: 22
- Run dir: eval_runs/backtests/20250831_000335

## Notes
- This is a proposal only; it does not modify live params.
- Apply manually after review or build a guarded canary flipper.
