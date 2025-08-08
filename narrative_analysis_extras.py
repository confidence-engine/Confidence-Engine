def adaptive_trigger(base_trigger: float, volume_z: float, min_trigger: float = 0.6, max_trigger: float = 1.5) -> float:
    """
    Adjust divergence trigger based on participation (volume_z).

    Intuition:
    - When participation is above average (volume_z > 0.5), slightly lower the trigger (more willing to act).
    - When participation is below average (volume_z < -0.5), slightly raise the trigger (more conservative).
    - Clamp final trigger between [min_trigger, max_trigger].

    Examples:
    - base_trigger=1.0, volume_z=+1.0 -> trigger ~0.8..0.9
    - base_trigger=1.0, volume_z=-1.0 -> trigger ~1.1..1.2
    """
    adj = 1.0
    if volume_z > 0.5:
        # Reduce up to ~20% as volume_z rises
        adj = max(0.8, 1.0 - min(0.2, (volume_z - 0.5) * 0.2))
    elif volume_z < -0.5:
        # Increase up to ~20% as volume_z falls
        adj = min(1.2, 1.0 + min(0.2, (-0.5 - volume_z) * 0.2))
    t = base_trigger * adj
    if t < min_trigger:
        t = min_trigger
    if t > max_trigger:
        t = max_trigger
    return float(t)
