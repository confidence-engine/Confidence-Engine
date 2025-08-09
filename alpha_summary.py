def alpha_summary(narrative_label: str, div: float, conf: float, price_label: str, vol_label: str, used_cnt: int) -> str:
    """
    Builds a trader-facing, alpha-first summary line.
    """
    tone = "Cautious"
    if abs(div) >= 1.0 and conf >= 0.60:
        tone = "Conviction"
    elif abs(div) >= 0.5 and conf >= 0.60:
        tone = "Actionable"
    elif abs(div) >= 0.3 and conf >= 0.55:
        tone = "Watchlist"

    return (
        f"Signal: {tone} — Story {narrative_label} vs Price {price_label} "
        f"(gap={div:+.2f}, conf={conf:.2f}, flow={vol_label}). "
        f"Evidence: {used_cnt} BTC headlines aligned; robust sentiment confirms {narrative_label.lower()} bias."
    )

def alpha_next_steps(div: float, conf: float, trig: float, vol_z: float) -> str:
    """
    Produces a concise playbook for next actions.
    """
    steps = []
    # Entry gating
    if abs(div) >= trig and conf >= 0.60 and vol_z > -0.5:
        direction = "LONG" if div > 0 else "SHORT"
        steps.append(f"Enter 1R {direction} on next 15m confirmation; scale to 1.5R if gap widens and conf ≥ 0.70.")
    else:
        steps.append("No entry: divergence below trigger or confidence insufficient.")

    # Monitoring and escalation
    steps.append("Set alert for |gap| ≥ trigger or volume_z > +0.7.")
    steps.append("Re-scan hourly; escalate if accepted ≥ 8 with same polarity and ≥ 2 distinct sources.")
    steps.append("Tighten sources to BTC-only if mixed-asset noise rises.")

    # Risk and invalidation
    steps.append("Risk 0.5–0.75R until confidence > 0.70 AND volume_z > 0.")
    steps.append("Invalidate if gap flips sign for 2 consecutive runs or FinBERT mean reverses by > 0.5.")

    return "Next steps:\n- " + "\n- ".join(steps)
