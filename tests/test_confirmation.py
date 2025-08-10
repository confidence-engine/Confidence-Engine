import os
from confirmation import run_confirmation_checks


def _mk_timescales(pct_short=0.6, vz_short=-0.5, vz_mid=-0.2, vz_long=-0.1, combined=0.35, align=False):
    return {
        "short": {"price_move_pct": pct_short, "divergence": 0.0, "volume_z": vz_short},
        "mid": {"price_move_pct": 0.0, "divergence": 0.0, "volume_z": vz_mid},
        "long": {"price_move_pct": 0.0, "divergence": 0.0, "volume_z": vz_long},
        "combined_divergence": combined,
        "alignment_flag": align,
    }


def test_confirmation_penalty_applied_defaults(monkeypatch):
    ts = _mk_timescales()
    env = {}
    checks, total = run_confirmation_checks(ts, env)
    # price_vs_narr (-0.02), volume_support (-0.01), timescale_alignment (-0.02) → -0.05 (clamped by min)
    assert round(total, 2) == -0.05
    assert any(c["name"] == "price_vs_narrative" and c["failed"] for c in checks)
    assert any(c["name"] == "volume_support" and c["failed"] for c in checks)
    assert any(c["name"] == "timescale_alignment" and c["failed"] for c in checks)


def test_confirmation_custom_env(monkeypatch):
    ts = _mk_timescales(pct_short=0.7, combined=0.4, align=False)
    env = {"TB_CONF_PRICE_VS_NARR": "-0.03", "TB_CONF_VOLUME_SUPPORT": "-0.02", "TB_CONF_TS_ALIGN": "-0.01", "TB_CONF_PENALTY_MIN": "-0.08"}
    checks, total = run_confirmation_checks(ts, env)
    assert round(total, 2) == -0.06
    names_failed = {c["name"] for c in checks if c["failed"]}
    assert {"price_vs_narrative", "volume_support", "timescale_alignment"}.issubset(names_failed)


def test_no_penalty_when_checks_pass():
    ts = _mk_timescales(pct_short=0.2, vz_short=0.1, vz_mid=0.1, vz_long=0.1, combined=0.1, align=True)
    env = {}
    checks, total = run_confirmation_checks(ts, env)
    assert total == 0.0
    assert not any(c["failed"] for c in checks)


