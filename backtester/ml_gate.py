import os
from typing import Optional

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from .features import build_features
from .models import EnhancedMLP


def predict_prob(bars: pd.DataFrame, model_path: str) -> Optional[float]:
    """Enhanced ML prediction with better error handling and model compatibility"""
    if not os.path.isfile(model_path):
        print(f"Warning: Model file not found at {model_path}")
        return None

    try:
        # Build features; take the last available sample
        X, _y = build_features(bars)
        if len(X) == 0:
            print("Warning: No features generated from bars data")
            return None

        x_np = X.iloc[[-1]].values.astype(float)

        # Replace any remaining NaN/Inf just in case
        x_np = np.where(np.isfinite(x_np), x_np, 0.0)

        x = torch.tensor(x_np, dtype=torch.float32)

        # Try to load model with enhanced architecture first, fallback to simple
        try:
            model = EnhancedMLP(in_dim=x.shape[1], hidden_dims=[64, 32, 16], dropout_rate=0.2)
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
            print(f"Successfully loaded enhanced model with {len(state)} parameters")
        except Exception as e:
            print(f"Enhanced model loading failed: {e}")
            print(f"State dict has {len(state)} keys: {list(state.keys())}")
            # Try to determine model architecture from state dict
            if 'net.0.weight' in state and len(state) > 6:
                print("Detected enhanced model architecture, trying again...")
                # Looks like enhanced model, try to load with correct architecture
                try:
                    model = EnhancedMLP(in_dim=x.shape[1], hidden_dims=[64, 32, 16], dropout_rate=0.2)
                    model.load_state_dict(state)
                    print("Successfully loaded enhanced model on second try")
                except Exception as e2:
                    print(f"Enhanced model loading failed again: {e2}")
                    return None
            else:
                print("Detected simple model architecture, using fallback...")
                # Fallback to simple architecture for backward compatibility
                try:
                    model = nn.Sequential(
                        nn.Linear(x.shape[1], 16),
                        nn.ReLU(),
                        nn.Linear(16, 1),
                    )
                    model.load_state_dict(state)
                    print("Successfully loaded simple model")
                except Exception as e3:
                    print(f"Simple model loading failed: {e3}")
                    return None

        model.eval()
        with torch.no_grad():
            logit = model(x)
            if not torch.isfinite(logit).all():
                print("Warning: Model produced non-finite logits")
                return None
            prob_t = torch.sigmoid(logit)
            if not torch.isfinite(prob_t).all():
                print("Warning: Model produced non-finite probabilities")
                return None
            prob = prob_t.item()
            if not np.isfinite(prob):
                print("Warning: Model produced non-finite probability value")
                return None

        return float(prob)

    except Exception as e:
        print(f"Error in ML prediction: {e}")
        return None


def get_model_info(model_path: str) -> dict:
    """Get information about the loaded model"""
    if not os.path.isfile(model_path):
        return {"error": "Model file not found"}

    try:
        # Load model to get architecture info
        dummy_input = torch.randn(1, 16)  # Dummy input to determine architecture
        model = EnhancedMLP(in_dim=16, hidden_dims=[64, 32, 16], dropout_rate=0.2)
        state = torch.load(model_path, map_location="cpu")

        return {
            "model_type": "EnhancedMLP",
            "parameters": sum(p.numel() for p in model.parameters()),
            "layers": len(list(model.parameters())),
            "loaded_successfully": True
        }
    except Exception as e:
        return {
            "model_type": "Unknown",
            "error": str(e),
            "loaded_successfully": False
        }
