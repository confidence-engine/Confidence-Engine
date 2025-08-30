# Analyzer ↔ Backtest Alignment

Latest replay: eval_runs/replays/20250830_231604

## Replay top configs (head)
(empty)

## Backtests top by Sharpe (head)
1. stop_mode=atr_trailing atr_mult=2.0 atr_period=21 tp_pct=0.03 time_cap_bars=0 risk_frac=0.01 Sharpe=-0.2789039425554448 CAGR=-0.0015015502700457262 max_dd=4.9815409468279585e-05
2. stop_mode=atr_trailing atr_mult=2.0 atr_period=21 tp_pct=0.03 time_cap_bars=0 risk_frac=0.01 Sharpe=-0.2789039425554448 CAGR=-0.0015015502700457262 max_dd=4.9815409468279585e-05
3. stop_mode=atr_trailing atr_mult=2.0 atr_period=21 tp_pct=0.01 time_cap_bars=0 risk_frac=0.01 Sharpe=-0.2789039425554448 CAGR=-0.0015015502700457262 max_dd=4.9815409468279585e-05
4. stop_mode=atr_trailing atr_mult=2.0 atr_period=21 tp_pct=0.01 time_cap_bars=0 risk_frac=0.01 Sharpe=-0.2789039425554448 CAGR=-0.0015015502700457262 max_dd=4.9815409468279585e-05
5. stop_mode=atr_trailing atr_mult=2.0 atr_period=21 tp_pct=0.02 time_cap_bars=0 risk_frac=0.01 Sharpe=-0.2789039425554448 CAGR=-0.0015015502700457262 max_dd=4.9815409468279585e-05
6. stop_mode=atr_trailing atr_mult=2.0 atr_period=21 tp_pct=0.02 time_cap_bars=0 risk_frac=0.01 Sharpe=-0.2789039425554448 CAGR=-0.0015015502700457262 max_dd=4.9815409468279585e-05
7. stop_mode=atr_trailing atr_mult=2.0 atr_period=21 tp_pct=0.03 time_cap_bars=0 risk_frac=0.02 Sharpe=-0.279020450980851 CAGR=-0.003000560933803853 max_dd=9.961715958720809e-05
8. stop_mode=atr_fixed atr_mult=2.0 atr_period=10 tp_pct=0.01 time_cap_bars=0 risk_frac=0.01 Sharpe=-0.2813554574622013 CAGR=-0.0017226917855827084 max_dd=8.665783901514797e-05
9. stop_mode=atr_fixed atr_mult=2.0 atr_period=10 tp_pct=0.02 time_cap_bars=0 risk_frac=0.01 Sharpe=-0.2813554574622013 CAGR=-0.0017226917855827084 max_dd=8.665783901514797e-05
10. stop_mode=atr_fixed atr_mult=2.0 atr_period=10 tp_pct=0.03 time_cap_bars=0 risk_frac=0.01 Sharpe=-0.2813554574622013 CAGR=-0.0017226917855827084 max_dd=8.665783901514797e-05

## Notes
- Compare replay ATR stop multiples and ML thresholds with backtest stop parameters.
- Prefer parameter regions that show strength in both offline replay cohorts and realized backtests.
