import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseTradingModel(ABC):
    """Abstract base class for trading models"""

    @abstractmethod
    def predict(self, features: np.ndarray) -> float:
        """Predict trading signal (-1 to 1)"""
        pass

    @abstractmethod
    def get_confidence(self, features: np.ndarray) -> float:
        """Get prediction confidence (0 to 1)"""
        pass

class EnhancedMLP(nn.Module):
    """Enhanced MLP with attention and regularization"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Attention mechanism
        attention_weights = F.softmax(self.attention(x), dim=0)
        attended_x = x * attention_weights

        return self.network(attended_x)

class AttentionMLP(BaseTradingModel):
    """MLP with attention mechanism for feature importance"""

    def __init__(self, input_dim: int):
        self.model = EnhancedMLP(input_dim)
        self.feature_importance = None

    def predict(self, features: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            output = self.model(x)
            prediction = torch.tanh(output).item()
            return prediction

    def get_confidence(self, features: np.ndarray) -> float:
        # Use prediction magnitude as confidence proxy
        prediction = abs(self.predict(features))
        return min(prediction * 2, 1.0)  # Scale to 0-1

class LSTMModel(nn.Module):
    """LSTM-based model for sequence prediction"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

class LSTMPredictor(BaseTradingModel):
    """LSTM-based trading predictor"""

    def __init__(self, input_dim: int, sequence_length: int = 20):
        self.model = LSTMModel(input_dim)
        self.sequence_length = sequence_length

    def predict(self, features: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            # Reshape features for sequence input
            if len(features.shape) == 1:
                # Single feature vector - repeat for sequence
                x = np.tile(features, (self.sequence_length, 1))
            else:
                x = features[-self.sequence_length:]  # Take last N periods

            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            output = self.model(x)
            prediction = torch.tanh(output).item()
            return prediction

    def get_confidence(self, features: np.ndarray) -> float:
        prediction = abs(self.predict(features))
        return min(prediction * 1.5, 1.0)

class TransformerBlock(nn.Module):
    """Transformer block for feature processing"""

    def __init__(self, embed_dim: int, num_heads: int = 8, ff_dim: int = 128):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x

class TransformerModel(nn.Module):
    """Transformer-based model"""

    def __init__(self, input_dim: int, num_blocks: int = 3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 64)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(64) for _ in range(num_blocks)
        ])
        self.output = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence dimension

        for block in self.transformer_blocks:
            x = block(x)

        x = x.squeeze(1)
        return self.output(x)

class TransformerPredictor(BaseTradingModel):
    """Transformer-based trading predictor"""

    def __init__(self, input_dim: int):
        self.model = TransformerModel(input_dim)

    def predict(self, features: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            output = self.model(x)
            prediction = torch.tanh(output).item()
            return prediction

    def get_confidence(self, features: np.ndarray) -> float:
        prediction = abs(self.predict(features))
        return min(prediction * 1.8, 1.0)

class CNNModel(nn.Module):
    """CNN-based model for pattern recognition"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Calculate flattened size
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class CNNPredictor(BaseTradingModel):
    """CNN-based trading predictor"""

    def __init__(self, input_dim: int):
        self.model = CNNModel(input_dim)

    def predict(self, features: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            output = self.model(x)
            prediction = torch.tanh(output).item()
            return prediction

    def get_confidence(self, features: np.ndarray) -> float:
        prediction = abs(self.predict(features))
        return min(prediction * 1.6, 1.0)

class MetaLearner(nn.Module):
    """Meta-learner to combine model predictions"""

    def __init__(self, num_models: int):
        super().__init__()
        self.weights = nn.Linear(num_models, num_models)
        self.output = nn.Linear(num_models, 1)

    def forward(self, predictions):
        # Learn optimal weights for combining predictions
        weights = F.softmax(self.weights(predictions), dim=-1)
        weighted_sum = (weights * predictions).sum(dim=-1, keepdim=True)
        return self.output(weighted_sum)

class TradingEnsemble:
    """
    Ensemble of specialized trading models with meta-learning
    """

    def __init__(self, input_dim: int):
        self.models = {
            'trend_model': AttentionMLP(input_dim),
            'mean_reversion_model': LSTMPredictor(input_dim),
            'breakout_model': CNNPredictor(input_dim),
            'momentum_model': TransformerPredictor(input_dim),
            'volatility_model': AttentionMLP(input_dim)
        }

        self.meta_learner = MetaLearner(len(self.models))
        self.model_weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        self.performance_history = []

    def predict(self, features: np.ndarray) -> float:
        """Generate ensemble prediction"""
        predictions = {}
        confidences = {}

        # Get predictions from all models
        for name, model in self.models.items():
            try:
                pred = model.predict(features)
                conf = model.get_confidence(features)
                predictions[name] = pred
                confidences[name] = conf
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")
                predictions[name] = 0.0
                confidences[name] = 0.0

        # Use meta-learner if trained, otherwise use weighted average
        if hasattr(self.meta_learner, 'state_dict'):
            try:
                pred_tensor = torch.tensor(list(predictions.values()), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    final_pred = self.meta_learner(pred_tensor).item()
                    return np.clip(final_pred, -1, 1)
            except Exception as e:
                logger.warning(f"Meta-learner failed: {e}")

        # Fallback to confidence-weighted average
        weighted_sum = 0.0
        total_weight = 0.0

        for name, pred in predictions.items():
            weight = confidences[name] * self.model_weights.get(name, 1.0)
            weighted_sum += pred * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0

    def get_ensemble_confidence(self, features: np.ndarray) -> float:
        """Get overall ensemble confidence"""
        predictions = []
        for model in self.models.values():
            try:
                pred = model.predict(features)
                predictions.append(pred)
            except Exception:
                predictions.append(0.0)

        if not predictions:
            return 0.0

        # Use prediction agreement as confidence measure
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        if std_pred == 0:
            return 1.0  # Perfect agreement

        # Confidence based on inverse of disagreement
        agreement = 1.0 / (1.0 + std_pred)
        return min(agreement, 1.0)

    def update_model_weights(self, recent_performance: Dict[str, float]):
        """Update model weights based on recent performance"""
        total_perf = sum(recent_performance.values())

        if total_perf > 0:
            for name in self.models.keys():
                perf = recent_performance.get(name, 0.0)
                self.model_weights[name] = perf / total_perf
        else:
            # Reset to equal weights if no performance data
            equal_weight = 1.0 / len(self.models)
            self.model_weights = {name: equal_weight for name in self.models.keys()}

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from attention models"""
        importance = {}

        for name, model in self.models.items():
            if hasattr(model, 'feature_importance') and model.feature_importance is not None:
                importance[name] = model.feature_importance

        return importance

class SpecializedModels:
    """Factory for creating specialized trading models"""

    @staticmethod
    def create_trend_model(input_dim: int) -> AttentionMLP:
        """Create model specialized for trend following"""
        return AttentionMLP(input_dim)

    @staticmethod
    def create_mean_reversion_model(input_dim: int) -> LSTMPredictor:
        """Create model specialized for mean reversion"""
        return LSTMPredictor(input_dim)

    @staticmethod
    def create_breakout_model(input_dim: int) -> CNNPredictor:
        """Create model specialized for breakout detection"""
        return CNNPredictor(input_dim)

    @staticmethod
    def create_momentum_model(input_dim: int) -> TransformerPredictor:
        """Create model specialized for momentum trading"""
        return TransformerPredictor(input_dim)

    @staticmethod
    def create_volatility_model(input_dim: int) -> AttentionMLP:
        """Create model specialized for volatility-based trading"""
        return AttentionMLP(input_dim)
