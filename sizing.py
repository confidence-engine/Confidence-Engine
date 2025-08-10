from __future__ import annotations

import os
from typing import Dict, Optional


def _f(env_key: str, default: float) -> float:
    try:
        return float(os.getenv(env_key, default))
    except Exception:
        return float(default)


def map_confidence_to_R(
    confidence: float,
    vol_norm: Optional[float] = None,
    cfg: Optional[Dict] = None,
) -> Dict:
    """
    Map final confidence to a target risk size (R) with floors/caps and optional volatility normalization.

    Env defaults (overridden by cfg if provided):
      - TB_SIZE_CONF_FLOOR=0.65
      - TB_SIZE_CONF_CAP=0.85
      - TB_SIZE_MAX_R=1.00
      - TB_SIZE_MIN_R=0.25
    """
    params = {
        "conf_floor": float((cfg or {}).get("conf_floor", _f("TB_SIZE_CONF_FLOOR", 0.65))),
        "conf_cap": float((cfg or {}).get("conf_cap", _f("TB_SIZE_CONF_CAP", 0.85))),
        "max_R": float((cfg or {}).get("max_R", _f("TB_SIZE_MAX_R", 1.00))),
        "min_R": float((cfg or {}).get("min_R", _f("TB_SIZE_MIN_R", 0.25))),
    }

    conf = float(confidence or 0.0)
    notes = []

    if conf < params["conf_floor"]:
        return {
            "confidence": conf,
            "target_R": 0.0,
            "notes": ["below floor"],
            "params": params,
        }

    # Linear map from [floor, cap] to [min_R, max_R]
    denom = params["conf_cap"] - params["conf_floor"]
    t = 1.0 if denom <= 0 else max(0.0, min(1.0, (conf - params["conf_floor"]) / denom))
    target_R = params["min_R"] + t * (params["max_R"] - params["min_R"])

    # Volatility normalization (optional)
    if vol_norm is not None:
        try:
            vn = float(vol_norm)
            if vn > 0:
                target_R = min(target_R / vn, params["max_R"])
                notes.append("vol_normalized")
        except Exception:
            pass

    # Clip to [0, max_R]
    target_R = max(0.0, min(params["max_R"], target_R))

    return {
        "confidence": conf,
        "target_R": float(round(target_R, 3)),
        "notes": notes,
        "params": params,
    }


