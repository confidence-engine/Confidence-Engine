# Backtest Comparison — Dynamic Stops (2025-08-30)

Runs compare stop modes with identical EMA parameters, fees, and TP, using `bars/`:

- fixed_pct: eval_runs/backtests/20250830_232739
- atr_fixed: eval_runs/backtests/20250830_232753
- atr_trailing: eval_runs/backtests/20250830_232811

Metrics snapshot:

- fixed_pct
  - cagr: -0.0038673087921130556
  - sharpe: -0.2975618683627886
  - max_dd: 0.00019079078785567475
  - win_rate: 0.36363636363636365
  - trades: 22
  - final_equity: 9998.896891146025

- atr_fixed
  - cagr: -0.005335353200350945
  - sharpe: -0.5791994836475746
  - max_dd: 0.00020378917212689443
  - win_rate: 0.22727272727272727
  - trades: 22
  - final_equity: 9998.477057218895

- atr_trailing
  - cagr: -0.006495726954382164
  - sharpe: -0.8691111826067325
  - max_dd: 0.00018552147024129226
  - win_rate: 0.18181818181818182
  - trades: 22
  - final_equity: 9998.144785297587

Notes:
- On this dataset/slice and parameters, fixed_pct performed best among the three.
- This is a narrow test; broader sweeps over `atr_mult`, different `tp_pct`, and different bars windows are recommended.
