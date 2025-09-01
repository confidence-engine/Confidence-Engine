import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn

from .ml_gate import predict_prob
from .ml_baseline import MarketRegimeDetector, DynamicRiskManager

logger = logging.getLogger(__name__)


class FeatureImportanceTracker:
    """Track and analyze feature importance over time"""

    def __init__(self):
        self.feature_importance_history = []
        self.feature_names = []
        self.importance_stats = defaultdict(list)

    def update_importance(self, feature_names: List[str], importance_scores: np.ndarray):
        """Update feature importance tracking"""
        self.feature_names = feature_names

        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'importance_scores': importance_scores.tolist(),
            'feature_names': feature_names
        }

        self.feature_importance_history.append(entry)

        # Update rolling statistics
        for i, name in enumerate(feature_names):
            self.importance_stats[name].append(importance_scores[i])

        # Keep only last 1000 entries
        if len(self.feature_importance_history) > 1000:
            self.feature_importance_history = self.feature_importance_history[-1000:]

        # Keep only last 100 importance values per feature
        for name in self.importance_stats:
            if len(self.importance_stats[name]) > 100:
                self.importance_stats[name] = self.importance_stats[name][-100:]

    def get_feature_stability(self) -> Dict[str, float]:
        """Calculate feature importance stability"""
        stability_scores = {}

        for feature, scores in self.importance_stats.items():
            if len(scores) >= 10:
                # Calculate coefficient of variation
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                cv = std_score / mean_score if mean_score > 0 else 0
                stability_scores[feature] = 1.0 / (1.0 + cv)  # Higher stability = lower CV
            else:
                stability_scores[feature] = 0.5  # Neutral for insufficient data

        return stability_scores

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features"""
        if not self.feature_importance_history:
            return []

        latest = self.feature_importance_history[-1]
        features_scores = list(zip(latest['feature_names'], latest['importance_scores']))

        # Sort by importance (descending)
        features_scores.sort(key=lambda x: x[1], reverse=True)

        return features_scores[:n]

    def detect_feature_drift(self) -> Dict[str, float]:
        """Detect drift in feature importance"""
        if len(self.feature_importance_history) < 20:
            return {'error': 'Insufficient data for drift detection'}

        # Compare recent vs older importance
        midpoint = len(self.feature_importance_history) // 2
        recent = self.feature_importance_history[midpoint:]
        older = self.feature_importance_history[:midpoint]

        def get_avg_importance(history_slice):
            if not history_slice:
                return {}
            all_scores = {}
            for entry in history_slice:
                for name, score in zip(entry['feature_names'], entry['importance_scores']):
                    if name not in all_scores:
                        all_scores[name] = []
                    all_scores[name].append(score)

            return {name: np.mean(scores) for name, scores in all_scores.items()}

        recent_avg = get_avg_importance(recent)
        older_avg = get_avg_importance(older)

        drift_scores = {}
        for feature in set(recent_avg.keys()) | set(older_avg.keys()):
            recent_score = recent_avg.get(feature, 0)
            older_score = older_avg.get(feature, 0)

            if older_score > 0:
                drift = abs(recent_score - older_score) / older_score
                drift_scores[feature] = drift
            else:
                drift_scores[feature] = 0.0

        return drift_scores


class AdvancedMLMonitor:
    """Advanced ML monitoring with comprehensive analytics"""

    def __init__(self, log_file: str = "ml_monitor.log"):
        self.log_file = log_file
        self.metrics_history = []
        self.prediction_history = []
        self.feature_tracker = FeatureImportanceTracker()
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = DynamicRiskManager()
        self.load_history()

    def load_history(self):
        """Load existing monitoring history"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.metrics_history = data.get('metrics_history', [])
                    self.prediction_history = data.get('prediction_history', [])
                    # Load feature importance if available
                    if 'feature_importance_history' in data:
                        self.feature_tracker.feature_importance_history = data['feature_importance_history']
                        self.feature_tracker.importance_stats = data.get('importance_stats', defaultdict(list))
            except Exception as e:
                print(f"Warning: Could not load ML monitor history: {e}")

    def save_history(self):
        """Save monitoring history"""
        data = {
            'metrics_history': self.metrics_history[-1000:],  # Keep last 1000 entries
            'prediction_history': self.prediction_history[-5000:],  # Keep last 5000 predictions
            'feature_importance_history': self.feature_tracker.feature_importance_history[-1000:],
            'importance_stats': dict(self.feature_tracker.importance_stats),
            'last_updated': datetime.utcnow().isoformat()
        }
        try:
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save ML monitor history: {e}")

    def log_prediction(self, bars: pd.DataFrame, probability: float, threshold: float,
                      decision: str, actual_outcome: Optional[bool] = None,
                      feature_importance: Optional[np.ndarray] = None,
                      feature_names: Optional[List[str]] = None):
        """Log a prediction with enhanced monitoring"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'probability': probability,
            'threshold': threshold,
            'decision': decision,  # 'pass', 'block', 'error'
            'bars_count': len(bars),
            'last_close': bars['close'].iloc[-1] if len(bars) > 0 else None,
            'actual_outcome': actual_outcome,  # True/False for correct prediction
            'features': self._extract_key_features(bars),
            'market_regime': self._detect_current_regime(bars),
            'risk_metrics': self._calculate_risk_metrics(bars)
        }

        self.prediction_history.append(entry)

        # Update feature importance tracking
        if feature_importance is not None and feature_names is not None:
            self.feature_tracker.update_importance(feature_names, feature_importance)

        self.save_history()

    def _detect_current_regime(self, bars: pd.DataFrame) -> Dict[str, str]:
        """Detect current market regime"""
        try:
            volatility_regime = self.regime_detector.detect_volatility_regime(bars)
            trend_regime = self.regime_detector.detect_trend_regime(bars)
            seasonal_regime = self.regime_detector.detect_seasonal_regime(datetime.utcnow())

            return {
                'volatility': volatility_regime,
                'trend': trend_regime,
                'seasonal': seasonal_regime
            }
        except Exception:
            return {'volatility': 'unknown', 'trend': 'unknown', 'seasonal': 'unknown'}

    def _calculate_risk_metrics(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics for the current position"""
        try:
            returns = bars['close'].pct_change().dropna()
            current_drawdown = self._calculate_drawdown(bars)
            var_estimate = self.risk_manager.estimate_var(returns)

            return {
                'current_drawdown': current_drawdown,
                'var_estimate': var_estimate,
                'volatility': returns.std(),
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0
            }
        except Exception:
            return {'current_drawdown': 0, 'var_estimate': 0.02, 'volatility': 0, 'sharpe_ratio': 0}

    def _calculate_drawdown(self, bars: pd.DataFrame) -> float:
        """Calculate current drawdown"""
        if len(bars) < 2:
            return 0.0

        peak = bars['close'].expanding().max()
        drawdown = (bars['close'] - peak) / peak
        return drawdown.iloc[-1]

    def get_regime_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by market regime"""
        regime_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'avg_prob': 0})

        for prediction in self.prediction_history:
            regime = prediction.get('market_regime', {})
            volatility_regime = regime.get('volatility', 'unknown')

            regime_stats[volatility_regime]['total'] += 1
            regime_stats[volatility_regime]['avg_prob'] += prediction['probability']

            if prediction.get('actual_outcome') is not None:
                regime_stats[volatility_regime]['correct'] += 1

        # Calculate percentages
        for regime, stats in regime_stats.items():
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']
                stats['avg_prob'] /= stats['total']

        return dict(regime_stats)

    def get_risk_adjusted_performance(self) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        if not self.prediction_history:
            return {'error': 'No prediction history'}

        # Extract returns and risk metrics
        returns = []
        volatilities = []
        drawdowns = []

        for prediction in self.prediction_history:
            risk_metrics = prediction.get('risk_metrics', {})
            if risk_metrics:
                volatilities.append(risk_metrics.get('volatility', 0))
                drawdowns.append(risk_metrics.get('current_drawdown', 0))

            # Calculate prediction returns (simplified)
            if prediction.get('actual_outcome') is not None:
                ret = 1 if prediction['actual_outcome'] else -1
                returns.append(ret)

        if not returns:
            return {'error': 'No outcome data available'}

        returns = np.array(returns)
        volatilities = np.array(volatilities) if volatilities else np.array([0.02])

        # Calculate risk-adjusted metrics
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0

        max_drawdown = min(drawdowns) if drawdowns else 0
        calmar_ratio = avg_return / abs(max_drawdown) if max_drawdown < 0 else 0

        return {
            'total_predictions': len(returns),
            'avg_return': avg_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': np.mean(returns > 0)
        }

    def get_feature_importance_analysis(self) -> Dict:
        """Get comprehensive feature importance analysis"""
        return {
            'top_features': self.feature_tracker.get_top_features(10),
            'feature_stability': self.feature_tracker.get_feature_stability(),
            'feature_drift': self.feature_tracker.detect_feature_drift(),
            'importance_trends': self._analyze_importance_trends()
        }

    def _analyze_importance_trends(self) -> Dict[str, List[float]]:
        """Analyze trends in feature importance over time"""
        if len(self.feature_tracker.feature_importance_history) < 10:
            return {}

        # Get last 10 importance snapshots
        recent_history = self.feature_tracker.feature_importance_history[-10:]

        trends = {}
        for feature in self.feature_tracker.feature_names:
            importance_series = []
            for entry in recent_history:
                if feature in entry['feature_names']:
                    idx = entry['feature_names'].index(feature)
                    importance_series.append(entry['importance_scores'][idx])
                else:
                    importance_series.append(0)

            trends[feature] = importance_series

        return trends

    def get_automated_retraining_signal(self) -> Dict[str, Any]:
        """Generate automated retraining signals based on multiple criteria"""

        signals = {
            'should_retrain': False,
            'reasons': [],
            'confidence': 0.0,
            'metrics': {}
        }

        # Check performance degradation
        health = self.get_model_health_score()
        if health.get('health_score', 1.0) < 0.6:
            signals['reasons'].append('Performance degradation detected')
            signals['confidence'] += 0.3

        # Check feature drift
        drift = self.feature_tracker.detect_feature_drift()
        avg_drift = np.mean(list(drift.values())) if drift else 0
        if avg_drift > 0.3:
            signals['reasons'].append('Significant feature drift detected')
            signals['confidence'] += 0.4

        # Check regime adaptation
        regime_perf = self.get_regime_performance()
        regime_accuracies = [stats.get('accuracy', 0) for stats in regime_perf.values()]
        if regime_accuracies and min(regime_accuracies) < 0.5:
            signals['reasons'].append('Poor performance in specific market regimes')
            signals['confidence'] += 0.3

        # Check prediction stability
        if len(self.prediction_history) > 100:
            recent_probs = [p['probability'] for p in self.prediction_history[-100:]]
            prob_std = np.std(recent_probs)
            if prob_std > 0.3:  # High variance in predictions
                signals['reasons'].append('Unstable prediction probabilities')
                signals['confidence'] += 0.2

        signals['should_retrain'] = signals['confidence'] >= 0.5
        signals['metrics'] = {
            'health_score': health.get('health_score', 0),
            'avg_drift': avg_drift,
            'min_regime_accuracy': min(regime_accuracies) if regime_accuracies else 0,
            'prediction_stability': 1.0 - min(1.0, prob_std) if 'prob_std' in locals() else 1.0
        }

        return signals

    def get_model_health_score(self) -> Dict[str, float]:
        """Calculate overall model health score with advanced metrics"""
        stats = self.get_performance_stats(hours=24)
        drift = self.feature_tracker.detect_feature_drift()
        regime_perf = self.get_regime_performance()
        risk_perf = self.get_risk_adjusted_performance()

        if 'error' in stats:
            return {'health_score': 0.0, 'status': 'insufficient_data'}

        # Calculate multiple health components
        health_components = {
            'prediction_stability': 1.0 - min(1.0, stats.get('error_rate', 0) * 2),
            'decision_balance': 1.0 - abs(stats.get('pass_rate', 0.5) - 0.5) * 2,
            'regime_adaptation': self._calculate_regime_adaptation_score(regime_perf),
            'risk_adjustment': self._calculate_risk_adjustment_score(risk_perf)
        }

        # Feature drift component
        drift_values = [v for v in drift.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        avg_drift = np.mean(drift_values) if drift_values else 0.0
        health_components['drift_score'] = 1.0 - min(1.0, avg_drift)

        health_score = np.mean(list(health_components.values()))

        status = 'healthy' if health_score > 0.7 else 'warning' if health_score > 0.5 else 'critical'

        return {
            'health_score': health_score,
            'status': status,
            'components': health_components,
            'recommendations': self._get_advanced_recommendations(health_score, stats, drift, regime_perf)
        }

    def _calculate_regime_adaptation_score(self, regime_perf: Dict) -> float:
        """Calculate how well the model adapts to different market regimes"""
        if not regime_perf:
            return 0.5

        accuracies = [stats.get('accuracy', 0) for stats in regime_perf.values() if stats.get('accuracy', 0) > 0]

        if not accuracies:
            return 0.5

        # Score based on consistency across regimes
        mean_accuracy = np.mean(accuracies)
        accuracy_std = np.std(accuracies)

        # Penalize high variance in regime performance
        consistency_score = 1.0 - min(1.0, accuracy_std * 2)

        return (mean_accuracy + consistency_score) / 2

    def _calculate_risk_adjustment_score(self, risk_perf: Dict) -> float:
        """Calculate risk-adjusted performance score"""
        if 'error' in risk_perf:
            return 0.5

        sharpe = risk_perf.get('sharpe_ratio', 0)
        calmar = risk_perf.get('calmar_ratio', 0)
        win_rate = risk_perf.get('win_rate', 0.5)

        # Combine risk-adjusted metrics
        risk_score = (sharpe + calmar + win_rate) / 3

        # Normalize to [0, 1]
        return max(0.0, min(1.0, risk_score))

    def _get_advanced_recommendations(self, health_score: float, stats: Dict, drift: Dict,
                                    regime_perf: Dict) -> List[str]:
        """Generate advanced recommendations based on comprehensive analysis"""
        recommendations = []

        if health_score < 0.7:
            if stats.get('error_rate', 0) > 0.1:
                recommendations.append("High error rate detected - check model loading and feature engineering")
            if abs(stats.get('pass_rate', 0.5) - 0.5) > 0.3:
                recommendations.append("Imbalanced pass/block rate - consider adjusting threshold")

        # Drift-specific recommendations
        drift_values = [v for v in drift.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        avg_drift = np.mean(drift_values) if drift_values else 0.0
        if avg_drift > 0.3:
            recommendations.append("High feature drift - consider model retraining with recent data")

        # Regime-specific recommendations
        if regime_perf:
            poor_regimes = [regime for regime, stats in regime_perf.items()
                          if stats.get('accuracy', 0) < 0.5]
            if poor_regimes:
                recommendations.append(f"Poor performance in regimes: {', '.join(poor_regimes)} - consider regime-specific models")

        if len(self.prediction_history) < 100:
            recommendations.append("Limited prediction history - continue monitoring for better insights")

        return recommendations if recommendations else ["Model performing well - continue monitoring"]

    def get_comprehensive_report(self) -> Dict:
        """Generate comprehensive ML monitoring report"""
        try:
            return {
                'accuracy': self.get_model_health_score().get('accuracy', 0.5),
                'drift_detected': self.get_automated_retraining_signal().get('should_retrain', False),
                'feature_importance': self.get_feature_importance_analysis(),
                'regime_performance': self.get_regime_performance(),
                'risk_adjusted_performance': self.get_risk_adjusted_performance(),
                'top_features': self.get_top_features(),
                'feature_stability': self.get_feature_stability(),
                'model_health': self.get_model_health_score(),
                'retraining_signal': self.get_automated_retraining_signal()
            }
        except Exception as e:
            logger.warning(f"Failed to generate comprehensive report: {e}")
            return {
                'accuracy': 0.5,
                'drift_detected': False,
                'error': str(e)
            }


# Enhanced global monitor instance
monitor = AdvancedMLMonitor()


def log_ml_prediction(bars: pd.DataFrame, probability: float, threshold: float,
                     decision: str, actual_outcome: Optional[bool] = None,
                     feature_importance: Optional[np.ndarray] = None,
                     feature_names: Optional[List[str]] = None):
    """Convenience function to log ML predictions with enhanced monitoring"""
    monitor.log_prediction(bars, probability, threshold, decision, actual_outcome,
                          feature_importance, feature_names)
    """Get comprehensive ML health report with advanced analytics"""
    return {
        'performance_stats': monitor.get_performance_stats(),
        'drift_analysis': monitor.detect_drift(),
        'health_score': monitor.get_model_health_score(),
        'regime_performance': monitor.get_regime_performance(),
        'risk_adjusted_performance': monitor.get_risk_adjusted_performance(),
        'feature_importance_analysis': monitor.get_feature_importance_analysis(),
        'retraining_signal': monitor.get_automated_retraining_signal(),
        'recent_predictions': monitor.prediction_history[-10:]  # Last 10 predictions
    }
