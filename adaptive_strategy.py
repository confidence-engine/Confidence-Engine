import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

# Optional imports with fallbacks
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Optuna not available, using fallback optimization")

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for strategy evaluation"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float

@dataclass
class StrategyParameters:
    """Dynamic strategy parameters"""
    risk_per_trade: float
    stop_loss_pct: float
    take_profit_pct: float
    max_positions: int
    sentiment_threshold: float
    ml_confidence_threshold: float
    regime_adjustments: Dict[str, float]

class PerformanceTracker:
    """Track and analyze trading performance"""

    def __init__(self):
        self.trades = []
        self.daily_returns = []
        self.peak_value = 1.0
        self.current_value = 1.0

    def add_trade(self, trade: Dict[str, Any]):
        """Add a completed trade to the tracker"""
        self.trades.append(trade)

        # Update portfolio value
        pnl = trade.get('pnl', 0)
        self.current_value *= (1 + pnl)

        # Update peak value for drawdown calculation
        self.peak_value = max(self.peak_value, self.current_value)

    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)

        # Basic metrics
        returns = [trade.get('pnl', 0) for trade in self.trades]
        cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0

        # Sharpe ratio
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        running_max = np.maximum.accumulate(1 + cumulative_returns)
        drawdowns = (running_max - (1 + cumulative_returns)) / running_max
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Win rate
        winning_trades = sum(1 for r in returns if r > 0)
        win_rate = winning_trades / len(returns) if returns else 0

        # Profit factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = [r for r in returns if r < 0]
        if downside_returns:
            downside_std = np.std(downside_returns)
            sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino_ratio = float('inf')

        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio
        )

    def analyze_recent_trades(self, lookback_periods: int = 50) -> Dict[str, Any]:
        """Analyze recent trading performance"""
        recent_trades = self.trades[-lookback_periods:] if len(self.trades) > lookback_periods else self.trades

        if not recent_trades:
            return {}

        analysis = {
            'total_trades': len(recent_trades),
            'win_rate': sum(1 for t in recent_trades if t.get('pnl', 0) > 0) / len(recent_trades),
            'avg_win': np.mean([t.get('pnl', 0) for t in recent_trades if t.get('pnl', 0) > 0]) if recent_trades else 0,
            'avg_loss': np.mean([t.get('pnl', 0) for t in recent_trades if t.get('pnl', 0) < 0]) if recent_trades else 0,
            'profit_factor': self._calculate_profit_factor(recent_trades),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(recent_trades),
            'volatility': np.std([t.get('pnl', 0) for t in recent_trades]) if recent_trades else 0
        }

        return analysis

    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor for a set of trades"""
        profits = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        losses = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))

        return profits / losses if losses > 0 else float('inf')

    def _calculate_max_consecutive_losses(self, trades: List[Dict[str, Any]]) -> int:
        """Calculate maximum consecutive losing trades"""
        max_consecutive = 0
        current_consecutive = 0

        for trade in trades:
            if trade.get('pnl', 0) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

class OnlineOptimizer:
    """Online parameter optimization using Bayesian optimization or fallback"""

    def __init__(self):
        if OPTUNA_AVAILABLE:
            self.study = optuna.create_study(direction='maximize')
        else:
            self.study = None
        self.parameter_history = []
        self.best_params = {}

    def optimize(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Optimize parameters based on recent performance"""

        if OPTUNA_AVAILABLE and self.study is not None:
            return self._optimize_with_optuna(performance_data)
        else:
            return self._optimize_fallback(performance_data)

    def _optimize_with_optuna(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Optimize using Optuna"""

        def objective(trial):
            # Define parameter search space
            params = {
                'risk_per_trade': trial.suggest_float('risk_per_trade', 0.005, 0.05),
                'stop_loss_pct': trial.suggest_float('stop_loss_pct', 0.01, 0.05),
                'take_profit_pct': trial.suggest_float('take_profit_pct', 0.02, 0.10),
                'sentiment_threshold': trial.suggest_float('sentiment_threshold', 0.3, 0.8),
                'ml_confidence_threshold': trial.suggest_float('ml_confidence_threshold', 0.5, 0.9)
            }

            # Simple objective based on win rate and profit factor
            win_rate = performance_data.get('win_rate', 0.5)
            profit_factor = performance_data.get('profit_factor', 1.0)

            # Penalize high volatility
            volatility_penalty = performance_data.get('volatility', 0) * 0.1

            score = win_rate * profit_factor - volatility_penalty

            return score

        # Run optimization
        self.study.optimize(objective, n_trials=20)

        # Get best parameters
        best_trial = self.study.best_trial
        optimized_params = best_trial.params

        # Store optimization results
        self.parameter_history.append({
            'timestamp': datetime.now(),
            'params': optimized_params,
            'score': best_trial.value,
            'performance_data': performance_data
        })

        self.best_params = optimized_params
        return optimized_params

    def _optimize_fallback(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Fallback optimization using simple rules"""

        # Get current performance
        win_rate = performance_data.get('win_rate', 0.5)
        volatility = performance_data.get('volatility', 0.02)

        # Simple rule-based optimization
        if win_rate > 0.6:
            # Performing well, can take more risk
            optimized_params = {
                'risk_per_trade': min(0.03, 0.02 * 1.2),
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.06,
                'sentiment_threshold': 0.45,
                'ml_confidence_threshold': 0.6
            }
        elif win_rate < 0.4:
            # Performing poorly, reduce risk
            optimized_params = {
                'risk_per_trade': max(0.01, 0.02 * 0.8),
                'stop_loss_pct': 0.025,
                'take_profit_pct': 0.04,
                'sentiment_threshold': 0.55,
                'ml_confidence_threshold': 0.7
            }
        else:
            # Average performance, maintain current levels
            optimized_params = {
                'risk_per_trade': 0.02,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05,
                'sentiment_threshold': 0.5,
                'ml_confidence_threshold': 0.65
            }

        # Adjust for volatility
        if volatility > 0.03:
            optimized_params['stop_loss_pct'] *= 1.2
            optimized_params['risk_per_trade'] *= 0.8
        elif volatility < 0.01:
            optimized_params['stop_loss_pct'] *= 0.8
            optimized_params['risk_per_trade'] *= 1.1

        # Store optimization results
        self.parameter_history.append({
            'timestamp': datetime.now(),
            'params': optimized_params,
            'score': win_rate,
            'performance_data': performance_data
        })

        self.best_params = optimized_params
        return optimized_params

class RegimeAdapter:
    """Adapt strategy parameters based on market regime"""

    def __init__(self):
        self.regime_performance = {}
        self.regime_parameters = self._initialize_regime_params()

    def _initialize_regime_params(self) -> Dict[str, StrategyParameters]:
        """Initialize regime-specific parameters"""
        return {
            'high_volatility': StrategyParameters(
                risk_per_trade=0.01,
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
                max_positions=2,
                sentiment_threshold=0.6,
                ml_confidence_threshold=0.7,
                regime_adjustments={'volatility_multiplier': 1.5}
            ),
            'low_volatility': StrategyParameters(
                risk_per_trade=0.025,
                stop_loss_pct=0.015,
                take_profit_pct=0.04,
                max_positions=4,
                sentiment_threshold=0.4,
                ml_confidence_threshold=0.6,
                regime_adjustments={'volatility_multiplier': 0.7}
            ),
            'trending': StrategyParameters(
                risk_per_trade=0.02,
                stop_loss_pct=0.02,
                take_profit_pct=0.05,
                max_positions=3,
                sentiment_threshold=0.5,
                ml_confidence_threshold=0.65,
                regime_adjustments={'trend_multiplier': 1.0}
            ),
            'sideways': StrategyParameters(
                risk_per_trade=0.015,
                stop_loss_pct=0.025,
                take_profit_pct=0.045,
                max_positions=2,
                sentiment_threshold=0.55,
                ml_confidence_threshold=0.7,
                regime_adjustments={'mean_reversion_multiplier': 1.2}
            ),
            'bull_market': StrategyParameters(
                risk_per_trade=0.025,
                stop_loss_pct=0.02,
                take_profit_pct=0.08,
                max_positions=4,
                sentiment_threshold=0.4,
                ml_confidence_threshold=0.6,
                regime_adjustments={'momentum_multiplier': 1.3}
            ),
            'bear_market': StrategyParameters(
                risk_per_trade=0.01,
                stop_loss_pct=0.025,
                take_profit_pct=0.04,
                max_positions=2,
                sentiment_threshold=0.65,
                ml_confidence_threshold=0.75,
                regime_adjustments={'defensive_multiplier': 1.5}
            )
        }

    def adapt_to_regime(self, regime: str, performance_data: Dict[str, Any]) -> StrategyParameters:
        """Adapt parameters based on current regime and performance"""
        base_params = self.regime_parameters.get(regime, self.regime_parameters['trending'])

        # Adjust based on recent performance
        if performance_data:
            win_rate = performance_data.get('win_rate', 0.5)

            # Increase risk if performing well, decrease if performing poorly
            if win_rate > 0.6:
                base_params.risk_per_trade = min(base_params.risk_per_trade * 1.1, 0.05)
            elif win_rate < 0.4:
                base_params.risk_per_trade = max(base_params.risk_per_trade * 0.9, 0.005)

            # Adjust stop loss based on volatility
            volatility = performance_data.get('volatility', 0)
            if volatility > 0.03:  # High volatility
                base_params.stop_loss_pct = min(base_params.stop_loss_pct * 1.2, 0.05)
            elif volatility < 0.01:  # Low volatility
                base_params.stop_loss_pct = max(base_params.stop_loss_pct * 0.8, 0.01)

        return base_params

    def update_regime_performance(self, regime: str, performance: PerformanceMetrics):
        """Update performance tracking for regime adaptation"""
        if regime not in self.regime_performance:
            self.regime_performance[regime] = []

        self.regime_performance[regime].append({
            'timestamp': datetime.now(),
            'metrics': performance
        })

        # Keep only recent performance data
        if len(self.regime_performance[regime]) > 100:
            self.regime_performance[regime] = self.regime_performance[regime][-100:]

class AdaptiveStrategy:
    """
    Complete adaptive strategy with real-time learning and parameter optimization
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/adaptive_strategy.json"
        self.performance_tracker = PerformanceTracker()
        self.parameter_optimizer = OnlineOptimizer()
        self.regime_adapter = RegimeAdapter()

        # Current strategy state
        self.current_regime = 'trending'
        self.current_parameters = self.regime_adapter.regime_parameters['trending']

        # Adaptation settings
        self.adaptation_interval = 24  # hours
        self.min_trades_for_adaptation = 10
        self.last_adaptation = datetime.now()

        # Load existing state if available
        self._load_state()

    def adapt_parameters(self, recent_performance: Dict[str, Any], current_regime: str = 'trending') -> Dict[str, Any]:
        """Main adaptation method called periodically"""

        # Check if enough time has passed for adaptation
        if (datetime.now() - self.last_adaptation).total_seconds() < self.adaptation_interval * 3600:
            return {'status': 'skipped', 'reason': 'too_soon'}

        # Check if we have enough data
        if len(self.performance_tracker.trades) < self.min_trades_for_adaptation:
            return {'status': 'skipped', 'reason': 'insufficient_data'}

        try:
            # Analyze recent performance
            analysis = self.performance_tracker.analyze_recent_trades()

            # Optimize parameters
            optimized_params = self.parameter_optimizer.optimize(analysis)

            # Adapt to current regime
            regime_params = self.regime_adapter.adapt_to_regime(current_regime, analysis)

            # Combine optimized and regime-specific parameters
            final_params = self._combine_parameters(optimized_params, regime_params)

            # Update current parameters
            self.current_parameters = final_params
            self.current_regime = current_regime
            self.last_adaptation = datetime.now()

            # Log adaptation
            adaptation_log = {
                'timestamp': datetime.now(),
                'regime': current_regime,
                'optimized_params': optimized_params,
                'regime_params': regime_params.__dict__,
                'final_params': final_params.__dict__,
                'performance_analysis': analysis
            }

            self._log_adaptation(adaptation_log)

            # Save state
            self._save_state()

            return {
                'status': 'adapted',
                'params': final_params.__dict__,
                'regime': current_regime,
                'analysis': analysis
            }

        except Exception as e:
            logger.error(f"Parameter adaptation failed: {e}")
            return {'status': 'error', 'reason': str(e)}

    def _combine_parameters(self, optimized: Dict[str, float], regime: StrategyParameters) -> StrategyParameters:
        """Combine optimized and regime-specific parameters"""

        # Use optimized parameters where available, otherwise use regime defaults
        combined = StrategyParameters(
            risk_per_trade=optimized.get('risk_per_trade', regime.risk_per_trade),
            stop_loss_pct=optimized.get('stop_loss_pct', regime.stop_loss_pct),
            take_profit_pct=optimized.get('take_profit_pct', regime.take_profit_pct),
            max_positions=regime.max_positions,  # Keep regime-specific
            sentiment_threshold=optimized.get('sentiment_threshold', regime.sentiment_threshold),
            ml_confidence_threshold=optimized.get('ml_confidence_threshold', regime.ml_confidence_threshold),
            regime_adjustments=regime.regime_adjustments
        )

        return combined

    def _log_adaptation(self, adaptation_data: Dict[str, Any]):
        """Log parameter adaptation for analysis"""
        try:
            log_path = Path("logs/adaptation_log.jsonl")
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with log_path.open('a') as f:
                json.dump(adaptation_data, f, default=str)
                f.write('\n')

        except Exception as e:
            logger.warning(f"Failed to log adaptation: {e}")

    def _save_state(self):
        """Save current strategy state"""
        try:
            state_path = Path("state/adaptive_strategy_state.json")
            state_path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                'current_regime': self.current_regime,
                'current_parameters': self.current_parameters.__dict__,
                'last_adaptation': self.last_adaptation.isoformat(),
                'performance_summary': self.performance_tracker.calculate_metrics().__dict__
            }

            with state_path.open('w') as f:
                json.dump(state, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to save strategy state: {e}")

    def _load_state(self):
        """Load previous strategy state"""
        try:
            state_path = Path("state/adaptive_strategy_state.json")
            if state_path.exists():
                with state_path.open('r') as f:
                    state = json.load(f)

                self.current_regime = state.get('current_regime', 'trending')
                self.last_adaptation = datetime.fromisoformat(state.get('last_adaptation', datetime.now().isoformat()))

                # Reconstruct parameters
                params_data = state.get('current_parameters', {})
                self.current_parameters = StrategyParameters(**params_data)

        except Exception as e:
            logger.warning(f"Failed to load strategy state: {e}")

    def get_current_parameters(self) -> StrategyParameters:
        """Get current strategy parameters"""
        return self.current_parameters

    def update_performance(self, trade_result: Dict[str, Any]):
        """Update performance tracker with new trade result"""
        self.performance_tracker.add_trade(trade_result)

        # Update regime performance
        self.regime_adapter.update_regime_performance(
            self.current_regime,
            self.performance_tracker.calculate_metrics()
        )

    def should_adapt(self) -> bool:
        """Check if strategy should adapt parameters"""
        time_since_adaptation = (datetime.now() - self.last_adaptation).total_seconds()
        has_enough_data = len(self.performance_tracker.trades) >= self.min_trades_for_adaptation

        return time_since_adaptation >= self.adaptation_interval * 3600 and has_enough_data
