import os
from datetime import datetime
from typing import Dict, Tuple

import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from .core import DataLoader
from .features import build_features


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_val_split(X: pd.DataFrame, y: pd.Series, val_frac: float = 0.2) -> Tuple:
    n = len(X)
    n_val = max(1, int(n * val_frac))
    X_train = X.iloc[:-n_val]
    y_train = y.iloc[:-n_val]
    X_val = X.iloc[-n_val:]
    y_val = y.iloc[-n_val:]
    return X_train, y_train, X_val, y_val


def metrics_from_logits(y_true: torch.Tensor, y_logit: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(y_logit).detach()
    preds = (probs >= 0.5).float()
    acc = (preds.squeeze() == y_true).float().mean().item()
    return {"accuracy": acc}


def run_ml_baseline(bars_dir: str, out_root: str = "eval_runs/ml/") -> str:
    loader = DataLoader(bars_dir)
    bars = loader.load()

    X, y = build_features(bars)
    if len(X) < 50:
        raise ValueError("Not enough data to train ML baseline")

    X_train, y_train, X_val, y_val = train_val_split(X, y)

    xtr = torch.tensor(X_train.values, dtype=torch.float32)
    ytr = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    xva = torch.tensor(X_val.values, dtype=torch.float32)
    yva = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)

    model = MLP(in_dim=xtr.shape[1])
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(200):
        opt.zero_grad()
        logits = model(xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(xva)
    metrics = metrics_from_logits(yva.squeeze(), val_logits.squeeze())

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"ml_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    pd.DataFrame(X.columns, columns=["feature"]).to_csv(os.path.join(out_dir, "features.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return out_dir
