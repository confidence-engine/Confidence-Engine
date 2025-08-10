from __future__ import annotations

from typing import Dict, List, Tuple


def _get(env: Dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except Exception:
        return float(default)


def run_confirmation_checks(timescales: Dict, env: Dict[str, str]) -> Tuple[List[Dict], float]:
    """
    Run negative-confirmation checks against timescale metrics.

    Returns (checks, total_penalty) where checks is a list of {name, failed, delta}.
    The total penalty is clamped to [TB_CONF_PENALTY_MIN, 0.0] (default [-0.05, 0]).
    """
    checks: List[Dict] = []

    # Extract needed fields safely
    short = timescales.get("short", {})
    mid = timescales.get("mid", {})
    long = timescales.get("long", {})
    combined_div = float(timescales.get("combined_divergence", 0.0) or 0.0)

    # price_vs_narrative
    price_vs_narr_delta = _get(env, "TB_CONF_PRICE_VS_NARR", -0.02)
    pv_failed = (abs(combined_div) > 0) and (float(short.get("price_move_pct", 0.0) or 0.0) > 0.5)
    checks.append({"name": "price_vs_narrative", "failed": bool(pv_failed), "delta": float(price_vs_narr_delta if pv_failed else 0.0)})

    # volume_support
    vol_delta = _get(env, "TB_CONF_VOLUME_SUPPORT", -0.01)
    avg_vz = (
        float(short.get("volume_z", 0.0) or 0.0)
        + float(mid.get("volume_z", 0.0) or 0.0)
        + float(long.get("volume_z", 0.0) or 0.0)
    ) / 3.0
    vol_failed = (abs(combined_div) >= 0.30) and (avg_vz < 0.0)
    checks.append({"name": "volume_support", "failed": bool(vol_failed), "delta": float(vol_delta if vol_failed else 0.0)})

    # timescale_alignment
    ts_delta = _get(env, "TB_CONF_TS_ALIGN", -0.02)
    alignment_flag = bool(timescales.get("alignment_flag", False))
    ts_failed = (not alignment_flag) and (abs(combined_div) >= 0.30)
    checks.append({"name": "timescale_alignment", "failed": bool(ts_failed), "delta": float(ts_delta if ts_failed else 0.0)})

    # Sum and clamp total penalty
    total = float(sum(chk["delta"] for chk in checks))
    min_total = _get(env, "TB_CONF_PENALTY_MIN", -0.05)
    if total < min_total:
        total = min_total
    if total > 0.0:
        total = 0.0
    return checks, total


