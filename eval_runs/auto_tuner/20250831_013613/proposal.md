# Auto-tuner Proposal (propose-only)

## Candidate config
- stop_mode: atr_trailing
- atr_mult: 2.0
- atr_period: 21
- tp_pct: 0.01
- sl_pct: 0.01
- time_cap_bars: 0
- risk_frac: 0.01

## Metrics
- Sharpe: -0.2789039425554448
- CAGR: -0.0015015502700457262
- Max DD: 4.9815409468279585e-05
- Trades: 22
- Run dir: eval_runs/backtests/20250830_235112

## Notes
- This is a proposal only; it does not modify live params.
- Apply manually after review or build a guarded canary flipper.
