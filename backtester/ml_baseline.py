import os
from datetime import datetime
from typing import Dict, Tuple

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from .core import DataLoader
from .features import build_features


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
