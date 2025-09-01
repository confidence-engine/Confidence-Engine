import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

from .ml_gate import predict_prob


class MLMonitor:
    """Monitor ML model performance and provide observability"""

    def __init__(self, log_file: str = "ml_monitor.log"):
        self.log_file = log_file
        self.metrics_history = []
        self.prediction_history = []
        self.load_history()

    def load_history(self):
        """Load existing monitoring history"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.metrics_history = data.get('metrics_history', [])
                    self.prediction_history = data.get('prediction_history', [])
            except Exception as e:
                print(f"Warning: Could not load ML monitor history: {e}")

    def save_history(self):
        """Save monitoring history"""
        data = {
            'metrics_history': self.metrics_history[-1000:],  # Keep last 1000 entries
            'prediction_history': self.prediction_history[-5000:],  # Keep last 5000 predictions
            'last_updated': datetime.utcnow().isoformat()
        }
        try:
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save ML monitor history: {e}")

    def log_prediction(self, bars: pd.DataFrame, probability: float, threshold: float,
                      decision: str, actual_outcome: Optional[bool] = None):
        """Log a prediction for monitoring"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'probability': probability,
            'threshold': threshold,
            'decision': decision,  # 'pass', 'block', 'error'
            'bars_count': len(bars),
            'last_close': bars['close'].iloc[-1] if len(bars) > 0 else None,
            'actual_outcome': actual_outcome,  # True/False for correct prediction
            'features': self._extract_key_features(bars)
        }

        self.prediction_history.append(entry)
        self.save_history()

    def log_metrics(self, metrics: Dict[str, float]):
        """Log model performance metrics"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics
        }

        self.metrics_history.append(entry)
        self.save_history()

    def _extract_key_features(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Extract key features for monitoring"""
        if len(bars) < 2:
            return {}

        try:
            close = bars['close']
            volume = bars.get('volume', pd.Series([1] * len(bars)))

            return {
                'current_price': float(close.iloc[-1]),
                'price_change_1': float(close.pct_change().iloc[-1]),
                'price_change_5': float(close.pct_change(periods=5).iloc[-1]) if len(close) >= 5 else 0,
                'volume_ratio': float(volume.iloc[-1] / volume.iloc[-2]) if len(volume) >= 2 else 1,
                'volatility': float(close.pct_change().rolling(20).std().iloc[-1]) if len(close) >= 20 else 0
            }
        except Exception:
            return {}

    def get_performance_stats(self, hours: int = 24) -> Dict[str, float]:
        """Get performance statistics for the last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        recent_predictions = [
            p for p in self.prediction_history
            if datetime.fromisoformat(p['timestamp']) > cutoff
        ]

        if not recent_predictions:
            return {'error': 'No recent predictions found'}

        stats = {
            'total_predictions': len(recent_predictions),
            'avg_probability': np.mean([p['probability'] for p in recent_predictions]),
            'pass_rate': np.mean([1 if p['decision'] == 'pass' else 0 for p in recent_predictions]),
            'block_rate': np.mean([1 if p['decision'] == 'block' else 0 for p in recent_predictions]),
            'error_rate': np.mean([1 if p['decision'] == 'error' else 0 for p in recent_predictions]),
        }

        # Accuracy if we have actual outcomes
        outcomes = [p for p in recent_predictions if p.get('actual_outcome') is not None]
        if outcomes:
            correct_predictions = sum(1 for p in outcomes if p['actual_outcome'])
            stats['accuracy'] = correct_predictions / len(outcomes)

        return stats

    def detect_drift(self) -> Dict[str, float]:
        """Detect potential feature drift"""
        if len(self.prediction_history) < 100:
            return {'error': 'Not enough data for drift detection'}

        # Compare recent vs older predictions
        midpoint = len(self.prediction_history) // 2
        recent = self.prediction_history[midpoint:]
        older = self.prediction_history[:midpoint]

        def get_feature_stats(predictions, feature_key):
            values = [p['features'].get(feature_key) for p in predictions if p['features'].get(feature_key) is not None]
            return np.mean(values), np.std(values) if values else (0, 0)

        drift_indicators = {}
        feature_keys = ['price_change_1', 'price_change_5', 'volume_ratio', 'volatility']

        for key in feature_keys:
            recent_mean, recent_std = get_feature_stats(recent, key)
            older_mean, older_std = get_feature_stats(older, key)

            if older_std > 0 and isinstance(recent_mean, (int, float)) and isinstance(older_mean, (int, float)):
                drift_indicators[f'{key}_drift'] = abs(recent_mean - older_mean) / older_std
            else:
                drift_indicators[f'{key}_drift'] = 0.0

        return drift_indicators

    def get_model_health_score(self) -> Dict[str, float]:
        """Calculate overall model health score"""
        stats = self.get_performance_stats(hours=24)
        drift = self.detect_drift()

        if 'error' in stats:
            return {'health_score': 0.0, 'status': 'insufficient_data'}

        # Calculate health score with safe averaging
        health_components = {
            'prediction_stability': 1.0 - min(1.0, stats.get('error_rate', 0) * 2),
            'decision_balance': 1.0 - abs(stats.get('pass_rate', 0.5) - 0.5) * 2,
        }

        # Safely calculate drift score
        drift_values = []
        for key, value in drift.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                drift_values.append(float(value))
            elif key == 'error':
                drift_values.append(1.0)  # High drift if error
            else:
                drift_values.append(0.0)

        if drift_values:
            avg_drift = np.mean(drift_values)
            health_components['drift_score'] = 1.0 - min(1.0, avg_drift)
        else:
            health_components['drift_score'] = 1.0

        health_score = np.mean(list(health_components.values()))

        status = 'healthy' if health_score > 0.7 else 'warning' if health_score > 0.5 else 'critical'

        return {
            'health_score': health_score,
            'status': status,
            'components': health_components,
            'recommendations': self._get_recommendations(health_score, stats, drift)
        }

    def _get_recommendations(self, health_score: float, stats: Dict, drift: Dict) -> List[str]:
        """Generate recommendations based on health analysis"""
        recommendations = []

        if health_score < 0.7:
            if stats.get('error_rate', 0) > 0.1:
                recommendations.append("High error rate detected - check model loading and feature engineering")
            if abs(stats.get('pass_rate', 0.5) - 0.5) > 0.3:
                recommendations.append("Imbalanced pass/block rate - consider adjusting threshold")
        # Safely check drift for recommendations
        drift_values = [v for v in drift.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        avg_drift = np.mean(drift_values) if drift_values else 0.0

        if avg_drift > 0.5:
            recommendations.append("Feature drift detected - consider model retraining")

        if len(self.prediction_history) < 100:
            recommendations.append("Limited prediction history - continue monitoring for better insights")

        return recommendations if recommendations else ["Model performing well - continue monitoring"]


# Global monitor instance
monitor = MLMonitor()


def log_ml_prediction(bars: pd.DataFrame, probability: float, threshold: float,
                     decision: str, actual_outcome: Optional[bool] = None):
    """Convenience function to log ML predictions"""
    monitor.log_prediction(bars, probability, threshold, decision, actual_outcome)


def get_ml_health_report() -> Dict:
    """Get comprehensive ML health report"""
    return {
        'performance_stats': monitor.get_performance_stats(),
        'drift_analysis': monitor.detect_drift(),
        'health_score': monitor.get_model_health_score(),
        'recent_predictions': monitor.prediction_history[-10:]  # Last 10 predictions
    }
