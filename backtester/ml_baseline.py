import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .core import DataLoader
from .features import build_features
from .models import EnhancedMLP, AttentionMLP, LSTMModel, TransformerModel, CNNModel, EnsembleModel
from .ml_gate import predict_prob


class AttentionLayer(nn.Module):
    """Multi-head attention mechanism for feature importance"""
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        # For single timestep, seq_len = 1
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + self.dropout(attn_output))


class EnhancedMLP(nn.Module):
    """Enhanced MLP with better architecture for trading predictions"""
    def __init__(self, in_dim: int, hidden_dims: list = [64, 32, 16], dropout_rate: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Final layer without dropout
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AttentionMLP(nn.Module):
    """MLP with attention mechanism for feature importance"""
    def __init__(self, in_dim: int, hidden_dims: list = [64, 32, 16], dropout_rate: float = 0.2):
        super().__init__()

        self.attention = AttentionLayer(in_dim)
        self.mlp = EnhancedMLP(in_dim, hidden_dims, dropout_rate)

    def forward(self, x):
        # Apply attention to input features
        x_attn = self.attention(x.unsqueeze(1))  # Add sequence dimension
        x_attn = x_attn.squeeze(1)  # Remove sequence dimension
        return self.mlp(x_attn)


class LSTMModel(nn.Module):
    """LSTM-based model for sequential trading patterns"""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # For single timestep, we need to expand
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # Take last timestep
        return self.fc(out)


class TransformerModel(nn.Module):
    """Transformer-based model for trading predictions"""
    def __init__(self, input_size: int, num_heads: int = 8, num_layers: int = 2, hidden_size: int = 64):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = self.input_projection(x).unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = self.dropout(x[:, -1, :])  # Take last timestep
        return self.fc(x)


class CNNModel(nn.Module):
    """CNN-based model for pattern recognition in trading data"""
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, input_size)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # Global average pooling
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for robust predictions"""
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.meta_learner = nn.Linear(len(models), 1)

    def forward(self, x):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)

        # Stack predictions
        stacked = torch.cat(predictions, dim=1)

        # Meta-learner combines predictions
        return self.meta_learner(stacked)


class MarketRegimeDetector:
    """Detect market regimes for adaptive trading"""
    def __init__(self):
        self.volatility_threshold = 0.02  # 2% daily volatility threshold
        self.trend_strength_threshold = 0.001  # Trend strength threshold

    def detect_volatility_regime(self, bars: pd.DataFrame) -> str:
        """Detect volatility regime: low, normal, high"""
        if len(bars) < 20:
            return "unknown"

        volatility = bars['close'].pct_change().rolling(20).std().iloc[-1]

        if volatility < self.volatility_threshold * 0.5:
            return "low_volatility"
        elif volatility > self.volatility_threshold * 2:
            return "high_volatility"
        else:
            return "normal_volatility"

    def detect_trend_regime(self, bars: pd.DataFrame) -> str:
        """Detect trend regime: strong_up, weak_up, sideways, weak_down, strong_down"""
        if len(bars) < 20:
            return "unknown"

        # Calculate trend strength using EMA slope
        ema_short = bars['close'].ewm(span=12).mean()
        ema_long = bars['close'].ewm(span=26).mean()
        trend_strength = (ema_short - ema_long).pct_change().abs().rolling(10).mean().iloc[-1]

        # Calculate trend direction
        macd = ema_short - ema_long
        trend_direction = "up" if macd.iloc[-1] > 0 else "down"

        if trend_strength > self.trend_strength_threshold * 2:
            return f"strong_{trend_direction}"
        elif trend_strength > self.trend_strength_threshold:
            return f"weak_{trend_direction}"
        else:
            return "sideways"

    def detect_seasonal_regime(self, current_time: datetime) -> str:
        """Detect seasonal patterns: morning, afternoon, evening, overnight"""
        hour = current_time.hour

        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 24:
            return "evening"
        else:
            return "overnight"


class DynamicRiskManager:
    """Advanced risk management with dynamic position sizing"""
    def __init__(self):
        self.max_drawdown_limit = 0.05  # 5% max drawdown
        self.var_limit = 0.02  # 2% VaR limit
        self.position_size_limits = {
            "low_volatility": 1.0,
            "normal_volatility": 0.7,
            "high_volatility": 0.3
        }

    def calculate_dynamic_position_size(self, confidence: float, volatility_regime: str,
                                      current_drawdown: float, var_estimate: float) -> float:
        """Calculate dynamic position size based on multiple factors"""

        # Base size from confidence
        base_size = confidence

        # Adjust for volatility regime
        regime_multiplier = self.position_size_limits.get(volatility_regime, 0.5)

        # Adjust for drawdown
        drawdown_multiplier = max(0.1, 1.0 - (current_drawdown / self.max_drawdown_limit))

        # Adjust for VaR
        var_multiplier = max(0.1, 1.0 - (var_estimate / self.var_limit))

        # Combine factors
        dynamic_size = base_size * regime_multiplier * drawdown_multiplier * var_multiplier

        return min(dynamic_size, 1.0)  # Cap at 100%

    def check_drawdown_limits(self, current_drawdown: float) -> bool:
        """Check if drawdown limits are breached"""
        return current_drawdown > self.max_drawdown_limit

    def estimate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Estimate Value at Risk"""
        if len(returns) < 30:
            return 0.02  # Default VaR

        return abs(np.percentile(returns, (1 - confidence_level) * 100))


class PortfolioOptimizer:
    """Portfolio optimization for multi-asset trading"""
    def __init__(self):
        self.max_assets = 5
        self.min_weight = 0.05
        self.max_weight = 0.4

    def optimize_weights(self, assets: List[str], expected_returns: Dict[str, float],
                        covariances: Dict[Tuple[str, str], float]) -> Dict[str, float]:
        """Optimize portfolio weights using risk-adjusted returns"""

        if len(assets) == 1:
            return {assets[0]: 1.0}

        # Simple risk-parity approach for now
        weights = {}
        total_weight = 0

        for asset in assets:
            # Weight inversely proportional to volatility (risk parity)
            volatility = np.sqrt(covariances.get((asset, asset), 0.0001))
            weight = 1.0 / volatility if volatility > 0 else 0.1
            weights[asset] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            weights = {asset: w / total_weight for asset, w in weights.items()}

        # Apply constraints
        weights = self._apply_constraints(weights)

        return weights

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply portfolio constraints"""
        # Ensure minimum and maximum weights
        constrained_weights = {}
        for asset, weight in weights.items():
            constrained_weights[asset] = max(self.min_weight, min(self.max_weight, weight))

        # Re-normalize
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {asset: w / total_weight for asset, w in constrained_weights.items()}

        return constrained_weights


class AdvancedMLTrainer:
    """Advanced ML trainer with multiple architectures and ensemble methods"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = DynamicRiskManager()
        self.portfolio_optimizer = PortfolioOptimizer()

    def train_ensemble_model(self, X: pd.DataFrame, y: pd.Series) -> EnsembleModel:
        """Train ensemble of multiple model architectures"""

        # Prepare data
        X_scaled = self.scaler.fit_transform(X.values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Train individual models
        models = []

        # 1. Enhanced MLP
        mlp_model = EnhancedMLP(X.shape[1])
        models.append(mlp_model)

        # 2. Attention MLP
        attention_model = AttentionMLP(X.shape[1])
        models.append(attention_model)

        # 3. LSTM (reshape for sequence)
        lstm_model = LSTMModel(X.shape[1])
        models.append(lstm_model)

        # 4. Transformer
        transformer_model = TransformerModel(X.shape[1])
        models.append(transformer_model)

        # 5. CNN
        cnn_model = CNNModel(X.shape[1])
        models.append(cnn_model)

        # Create ensemble
        ensemble = EnsembleModel(models)

        # Train ensemble (simplified training for demonstration)
        optimizer = optim.Adam(ensemble.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        # Simple training loop
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = ensemble(X_tensor)
            loss = criterion(outputs.squeeze(), torch.tensor(y.values, dtype=torch.float32))
            loss.backward()
            optimizer.step()

        return ensemble

    def train_sklearn_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train sklearn ensemble models for comparison"""

        X_scaled = self.scaler.fit_transform(X.values)

        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42)
        }

        trained_models = {}
        for name, model in models.items():
            model.fit(X_scaled, y.values)
            trained_models[name] = model

        return trained_models

    def predict_with_regime_adaptation(self, bars: pd.DataFrame, model_path: str) -> Dict:
        """Make predictions with regime-aware adjustments"""

        # Detect current market regime
        volatility_regime = self.regime_detector.detect_volatility_regime(bars)
        trend_regime = self.regime_detector.detect_trend_regime(bars)
        seasonal_regime = self.regime_detector.detect_seasonal_regime(datetime.utcnow())

        # Get base ML prediction
        base_prob = predict_prob(bars, model_path) or 0.5

        # Apply regime adjustments
        regime_adjustments = {
            "low_volatility": 0.1,  # Increase confidence in low vol
            "high_volatility": -0.1,  # Decrease confidence in high vol
            "normal_volatility": 0.0
        }

        trend_adjustments = {
            "strong_up": 0.05,
            "strong_down": -0.05,
            "weak_up": 0.02,
            "weak_down": -0.02,
            "sideways": 0.0
        }

        seasonal_adjustments = {
            "morning": 0.02,  # Slightly bullish morning
            "afternoon": 0.0,
            "evening": -0.01,
            "overnight": -0.02
        }

        adjusted_prob = base_prob
        adjusted_prob += regime_adjustments.get(volatility_regime, 0.0)
        adjusted_prob += trend_adjustments.get(trend_regime, 0.0)
        adjusted_prob += seasonal_adjustments.get(seasonal_regime, 0.0)

        # Clip to [0, 1]
        adjusted_prob = max(0.0, min(1.0, adjusted_prob))

        return {
            'base_probability': base_prob,
            'adjusted_probability': adjusted_prob,
            'volatility_regime': volatility_regime,
            'trend_regime': trend_regime,
            'seasonal_regime': seasonal_regime,
            'regime_adjustment': adjusted_prob - base_prob
        }


# Global trainer instance
trainer = AdvancedMLTrainer()


class EnhancedMLP(nn.Module):
    """Enhanced MLP with better architecture for trading predictions"""
    def __init__(self, in_dim: int, hidden_dims: list = [64, 32, 16], dropout_rate: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Final layer without dropout
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_val_split(X: pd.DataFrame, y: pd.Series, val_frac: float = 0.2) -> Tuple:
    n = len(X)
    n_val = max(1, int(n * val_frac))
    X_train = X.iloc[:-n_val]
    y_train = y.iloc[:-n_val]
    X_val = X.iloc[-n_val:]
    y_val = y.iloc[-n_val:]
    return X_train, y_train, X_val, y_val


def metrics_from_logits(y_true: torch.Tensor, y_logit: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(y_logit).detach().cpu().numpy()
    preds = (probs >= 0.5).astype(int).flatten()
    y_true_np = y_true.cpu().numpy().astype(int)

    metrics = {
        "accuracy": float(np.mean(preds == y_true_np)),
        "precision": float(precision_score(y_true_np, preds, zero_division=0)),
        "recall": float(recall_score(y_true_np, preds, zero_division=0)),
        "f1_score": float(f1_score(y_true_np, preds, zero_division=0)),
    }

    # Add AUC if we have both classes
    if len(np.unique(y_true_np)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true_np, probs.flatten()))

    return metrics


def run_ml_baseline(bars_dir: str, out_root: str = "eval_runs/ml/") -> str:
    loader = DataLoader(bars_dir)
    bars = loader.load()

    X, y = build_features(bars)
    if len(X) < 100:  # Need more data for enhanced model
        raise ValueError("Not enough data to train enhanced ML model (need at least 100 samples)")

    X_train, y_train, X_val, y_val = train_val_split(X, y, val_frac=0.25)

    xtr = torch.tensor(X_train.values, dtype=torch.float32)
    ytr = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    xva = torch.tensor(X_val.values, dtype=torch.float32)
    yva = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)

    # Enhanced model with better architecture
    model = EnhancedMLP(in_dim=xtr.shape[1], hidden_dims=[64, 32, 16], dropout_rate=0.2)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    loss_fn = nn.BCEWithLogitsLoss()

    # Early stopping
    early_stopping = EarlyStopping(patience=30, min_delta=0.001)

    model.train()
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(500):  # More epochs with early stopping
        # Training step
        opt.zero_grad()
        logits = model(xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_logits = model(xva)
            val_loss = loss_fn(val_logits, yva).item()
        model.train()

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

    # Load best model for final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(xva)
    metrics = metrics_from_logits(yva.squeeze(), val_logits.squeeze())

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"ml_enhanced_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    pd.DataFrame(X.columns, columns=["feature"]).to_csv(os.path.join(out_dir, "features.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training metadata
    metadata = {
        "model_type": "EnhancedMLP",
        "input_dim": xtr.shape[1],
        "hidden_dims": [64, 32, 16],
        "dropout_rate": 0.2,
        "training_samples": len(xtr),
        "validation_samples": len(xva),
        "features_count": len(X.columns),
        "feature_names": list(X.columns)
    }

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return out_dir


# Backward compatibility
MLP = EnhancedMLP
