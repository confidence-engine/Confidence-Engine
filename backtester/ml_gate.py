import os
from typing import Optional

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from .features import build_features
from .ml_baseline import MLP


def predict_prob(bars: pd.DataFrame, model_path: str) -> Optional[float]:
    if not os.path.isfile(model_path):
        return None
    # Build features; take the last available sample
    X, _y = build_features(bars)
    if len(X) == 0:
        return None
    x_np = X.iloc[[-1]].values.astype(float)
    # Replace any remaining NaN/Inf just in case
    x_np = np.where(np.isfinite(x_np), x_np, 0.0)
    x = torch.tensor(x_np, dtype=torch.float32)
    model = MLP(in_dim=x.shape[1])
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logit = model(x)
        if not torch.isfinite(logit).all():
            return None
        prob_t = torch.sigmoid(logit)
        if not torch.isfinite(prob_t).all():
            return None
        prob = prob_t.item()
        if not np.isfinite(prob):
            return None
    return float(prob)
