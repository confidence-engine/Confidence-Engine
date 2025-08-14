"""
Evidence line generator for surfaced assets.

Rules:
- Inputs: sentiment tag, participation tag, TF alignment boolean, signal quality tag, narrative tags.
- Output: 1–2 sentences, plain English, no numeric values.
- Provide a helper to strip numbers for chat outputs (TG/Discord), keeping artifacts untouched.
"""
from typing import List, Dict, Any, Optional
import re
import os

def estimate_confidence_pct(signal_quality_tag: str, tf_aligned: bool, risk_band: str) -> float:
    """
    Lightweight agent confidence estimator in [0.0, 1.0].
    Rules of thumb:
    - signal_quality: low≈0.4, moderate≈0.55, strong/high≈0.7, very high≈0.85
    - +0.10 if tf_aligned
    - Risk penalty: risk_band high -> -0.05, extreme -> -0.10
    - Clamp to [0.2, 0.98]
    """
    sq = (signal_quality_tag or "").strip().lower()
    if ("very" in sq and "high" in sq) or ("vh" in sq):
        base = 0.85
    elif "strong" in sq or "high" in sq:
        base = 0.70
    elif "mod" in sq or "medium" in sq or "moderate" in sq:
        base = 0.55
    elif "weak" in sq or "low" in sq:
        base = 0.40
    else:
        base = 0.50
    if bool(tf_aligned):
        base += 0.10
    rb = (risk_band or "").strip().lower()
    if "extreme" in rb:
        base -= 0.10
    elif "high" in rb:
        base -= 0.05
    return max(0.20, min(0.98, base))


def generate_high_risk_note(risk_band: str, action: str, risk_score: float) -> str:
    """
    Generate a graded risk explanation when risk is high and action is Buy/Watch.
    Returns empty string otherwise.
    """
    rb = (risk_band or "").strip().lower()
    act = (action or "").strip().title()
    try:
        rs = float(risk_score)
    except Exception:
        rs = 0.0
    if rb != "high" or act not in ("Buy", "Watch"):
        return ""
    if rs >= 0.85:
        level = "extreme"
    elif rs >= 0.7:
        level = "very high"
    else:
        level = "elevated"
    if act == "Buy":
        return f"{level.capitalize()}-risk long setup; size aggressively low and expect wide swings."
    else:
        return f"{level.capitalize()}-risk potential; only suitable if comfortable with elevated volatility."


def strip_numbers_for_chat(text: str) -> str:
    """
    Remove numeric values from a string for safe chat rendering.
    Preserves letters and punctuation, collapses extra whitespace.
    """
    if not text:
        return ""
    # Remove digits and common numeric formats (including decimals, percents, +/-, commas)
    cleaned = re.sub(r"[\d]+(?:[\.,][\d]+)?%?", "", text)
    # Remove stray plus/minus signs attached to numbers we just stripped
    cleaned = re.sub(r"[\+\-]\s*", "", cleaned)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _choose_sentiment_phrase(sentiment: str) -> str:
    s = (sentiment or "").strip().lower()
    if "bull" in s or s in {"buy", "long"}:
        return "Price looks bullish"
    if "bear" in s or s in {"sell", "short"}:
        return "Price looks bearish"
    if "watch" in s or "neutral" in s:
        return "Price looks mixed"
    return "Price is developing"


def _choose_confidence_phrase(signal_quality: str, tf_aligned: bool) -> str:
    q = (signal_quality or "").strip().lower()
    very_high = ("very" in q and "high" in q) or ("vh" in q)
    high = ("strong" in q or "high" in q)
    if very_high and tf_aligned:
        return "with very high confidence"
    if very_high and not tf_aligned:
        # Dominant TF leads; minor divergences present
        return "with very high confidence; dominant timeframe leads; minor divergences present"
    if tf_aligned and high:
        return "with high confidence"
    if tf_aligned:
        return "with moderate confidence"
    if "weak" in q or "low" in q:
        return "with low confidence"
    return "with mixed confidence"


def _choose_participation_phrase(participation: str) -> str:
    p = (participation or "").strip().lower()
    if "hot" in p or "elevated" in p or "active" in p:
        return "Trading activity is busy"
    if "quiet" in p or "thin" in p:
        return "Trading activity is quiet"
    return "Trading activity is normal"


def _choose_narrative_phrase(narrative_tags: List[str]) -> str:
    tags = " ".join((narrative_tags or [])).lower()
    if "continuation" in tags or "trend" in tags:
        return "and the pattern supports continuation"
    if "reversion" in tags or "fade" in tags:
        return "and the pattern favors mean reversion"
    if "breakout" in tags or "breakdown" in tags:
        return "and the pattern favors breakouts"
    return "and the pattern is well-defined"


def generate_evidence_line(
    sentiment_tag: str,
    participation_tag: str,
    tf_aligned: bool,
    signal_quality_tag: str,
    narrative_tags: List[str],
) -> str:
    """
    Generate a concise, number‑free "why now" string.

    Returns up to 2 sentences, no numeric values.
    """
    # Sentence 1: Sentiment + confidence
    s1 = f"{_choose_sentiment_phrase(sentiment_tag)} {_choose_confidence_phrase(signal_quality_tag, tf_aligned)}."
    # Sentence 2: Participation + narrative
    s2 = f"{_choose_participation_phrase(participation_tag)}, {_choose_narrative_phrase(narrative_tags)}."

    out = f"{s1} {s2}".strip()
    # Ensure no numbers, even if user passed numeric-like tags
    out = strip_numbers_for_chat(out)
    return out


# --- Shared grading helpers (letter grades A+..D) ---
def _infer_signal_quality(asset: Dict[str, Any], th: Dict[str, Any], aligned: bool) -> str:
    """
    Prefer explicit `signal_quality`; else infer from `confirmation_checks`.
    Returns one of {'strong','elevated','mixed','low'}.
    """
    sigq = (asset.get("signal_quality") or th.get("signal_quality") or "").strip().lower()
    if sigq:
        return sigq
    checks = asset.get("confirmation_checks") or []
    try:
        passed = {str(c.get("name")): bool(c.get("passed")) for c in checks if isinstance(c, dict)}
        pv = passed.get("price_vs_narrative", False)
        vol = passed.get("volume_support", False)
        tfa = passed.get("timescale_alignment", False)
        if pv and vol and (tfa or aligned):
            return "strong"
        if pv or vol:
            return "elevated"
        return "mixed"
    except Exception:
        return "mixed"


def _aplus_condition(asset: Dict[str, Any]) -> bool:
    """Mirrors the strict A+ setup heuristic used by chat formatters."""
    if not isinstance(asset, dict):
        return False
    th = asset.get("thesis") or {}
    action = (th.get("action") or asset.get("action") or "").strip().lower()
    if action not in ("buy", "long", "sell", "short"):
        return False
    readiness = (th.get("readiness") or asset.get("readiness") or "").strip().lower()
    if readiness not in ("now", "near"):
        return False
    tss = asset.get("timescale_scores") or {}
    aligned = bool(
        asset.get("alignment_flag")
        or th.get("tf_aligned")
        or (isinstance(tss, dict) and (tss.get("alignment_flag") or (tss.get("aligned_horizons", 0) or 0) >= 2))
    )
    if not aligned:
        return False
    sigq = _infer_signal_quality(asset, th, aligned)
    if sigq not in ("strong", "elevated"):
        return False
    risk_band = (th.get("risk_band") or asset.get("risk") or asset.get("risk_band") or "").strip().lower()
    if risk_band == "high":
        return False
    return True


def compute_setup_grade(asset: Dict[str, Any]) -> str:
    """
    Compute a letter grade in {A+, A, B, C, D} using agent-facing fields only.
    If the strict A+ heuristic passes, upgrade an A grade to A+.
    """
    try:
        th = asset.get("thesis") or {}
        tss = asset.get("timescale_scores") or {}
        aligned = bool(
            asset.get("alignment_flag")
            or th.get("tf_aligned")
            or (isinstance(tss, dict) and (tss.get("alignment_flag") or (tss.get("aligned_horizons", 0) or 0) >= 2))
        )
        rb = (th.get("risk_band") or asset.get("risk") or asset.get("risk_band") or "").strip().lower()
        sigq = _infer_signal_quality(asset, th, aligned)
        # Estimated agent confidence [0..1]
        conf = estimate_confidence_pct(sigq, aligned, rb)
        # Confirmation score from checks
        checks = asset.get("confirmation_checks") or []
        passed = {str(c.get("name")): bool(c.get("passed")) for c in checks if isinstance(c, dict)}
        pv = passed.get("price_vs_narrative", False)
        vol = passed.get("volume_support", False)
        tfa = passed.get("timescale_alignment", False)
        if pv and vol and (tfa or aligned):
            cscore = 1.0
        elif pv or vol:
            cscore = 0.66
        else:
            cscore = 0.33
        ascore = 1.0 if aligned else 0.0
        # Weights (tunable via env)
        try:
            w_conf = float(os.getenv("TB_GRADE_W_CONF", "0.5"))
            w_chk = float(os.getenv("TB_GRADE_W_CONFCHK", "0.3"))
            w_aln = float(os.getenv("TB_GRADE_W_ALIGN", "0.2"))
        except Exception:
            w_conf, w_chk, w_aln = 0.5, 0.3, 0.2
        denom = max(1e-6, (w_conf + w_chk + w_aln))
        w_conf, w_chk, w_aln = w_conf/denom, w_chk/denom, w_aln/denom
        score = w_conf * conf + w_chk * cscore + w_aln * ascore
        # Thresholds (tunable via env). Ensure order: A>=ta, B>=tb, C>=tc
        try:
            ta = float(os.getenv("TB_GRADE_THRESH_A", "0.80"))
            tb = float(os.getenv("TB_GRADE_THRESH_B", "0.65"))
            tc = float(os.getenv("TB_GRADE_THRESH_C", "0.50"))
        except Exception:
            ta, tb, tc = 0.80, 0.65, 0.50
        if score >= ta:
            grade = "A"
        elif score >= tb:
            grade = "B"
        elif score >= tc:
            grade = "C"
        else:
            grade = "D"
        if grade == "A" and _aplus_condition(asset):
            return "A+"
        return grade
    except Exception:
        return "C"


def compute_setup_grade_for_tf(asset: Dict[str, Any], tf_key: str, plan_tf: Optional[Dict[str, Any]] = None) -> str:
    """
    Compute a per-timeframe letter grade {A+, A, B, C, D} using the same inputs as
    compute_setup_grade() plus a small TF-local adjustment based on plan provenance.

    Adjustments:
    - source == 'analysis' -> +0.03 to score (agent-mode plan present)
    - source == 'fallback' -> -0.04 to score (heuristic plan)
    - presence of a non-empty 'explain' -> +0.01 (better rationale)
    """
    try:
        th = asset.get("thesis") or {}
        tss = asset.get("timescale_scores") or {}
        aligned = bool(
            asset.get("alignment_flag")
            or th.get("tf_aligned")
            or (isinstance(tss, dict) and (tss.get("alignment_flag") or (tss.get("aligned_horizons", 0) or 0) >= 2))
        )
        rb = (th.get("risk_band") or asset.get("risk") or asset.get("risk_band") or "").strip().lower()
        sigq = _infer_signal_quality(asset, th, aligned)
        # Base estimated agent confidence
        conf = estimate_confidence_pct(sigq, aligned, rb)
        # Confirmation score from checks
        checks = asset.get("confirmation_checks") or []
        passed = {str(c.get("name")): bool(c.get("passed")) for c in checks if isinstance(c, dict)}
        pv = passed.get("price_vs_narrative", False)
        vol = passed.get("volume_support", False)
        tfa = passed.get("timescale_alignment", False)
        if pv and vol and (tfa or aligned):
            cscore = 1.0
        elif pv or vol:
            cscore = 0.66
        else:
            cscore = 0.33
        ascore = 1.0 if aligned else 0.0
        # Weights (env tunable)
        try:
            w_conf = float(os.getenv("TB_GRADE_W_CONF", "0.5"))
            w_chk = float(os.getenv("TB_GRADE_W_CONFCHK", "0.3"))
            w_aln = float(os.getenv("TB_GRADE_W_ALIGN", "0.2"))
        except Exception:
            w_conf, w_chk, w_aln = 0.5, 0.3, 0.2
        denom = max(1e-6, (w_conf + w_chk + w_aln))
        w_conf, w_chk, w_aln = w_conf/denom, w_chk/denom, w_aln/denom
        score = w_conf * conf + w_chk * cscore + w_aln * ascore
        # TF-local adjustment from plan provenance
        p = plan_tf if isinstance(plan_tf, dict) else ((asset.get("plan") or {}).get(tf_key) or {})
        src = (p.get("source") or "").strip().lower()
        if src == "analysis":
            score += 0.03
        elif src == "fallback":
            score -= 0.04
        try:
            if isinstance(p.get("explain"), str) and p.get("explain").strip():
                score += 0.01
        except Exception:
            pass
        # Clamp
        score = max(0.0, min(1.0, score))
        # Thresholds (env tunable)
        try:
            ta = float(os.getenv("TB_GRADE_THRESH_A", "0.80"))
            tb = float(os.getenv("TB_GRADE_THRESH_B", "0.65"))
            tc = float(os.getenv("TB_GRADE_THRESH_C", "0.50"))
        except Exception:
            ta, tb, tc = 0.80, 0.65, 0.50
        if score >= ta:
            grade = "A"
        elif score >= tb:
            grade = "B"
        elif score >= tc:
            grade = "C"
        else:
            grade = "D"
        if grade == "A" and _aplus_condition(asset):
            return "A+"
        return grade
    except Exception:
        return "C"


def compute_polymarket_grade(pm: Dict[str, Any]) -> str:
    """
    Lightweight grade for Polymarket items where we have fewer fields.
    Signals used: stance/readiness, edge_label (rich/cheap), optional internal_prob.
    """
    try:
        stance = (pm.get("stance") or "").strip().lower()
        readiness = (pm.get("readiness") or "").strip().lower()
        edge = (pm.get("edge_label") or pm.get("implied_side") or "").strip().lower()
        ip = pm.get("internal_prob")
        # Base confidence
        conf = 0.5
        if isinstance(ip, (int, float)):
            # map [0,1] roughly if it's 0..1, else if 0..100 convert
            v = float(ip)
            conf = v if 0.0 <= v <= 1.0 else max(0.0, min(1.0, v / 100.0))
        # Edge boost
        edge_boost = 0.15 if ("rich" in edge or "cheap" in edge) else 0.0
        # Readiness boost
        ready_boost = 0.15 if readiness in ("now", "near") else 0.0
        # Stance weight (Engage/Stalk > Stand Aside)
        stance_boost = 0.1 if any(k in stance for k in ("engage", "stalk")) else 0.0
        score = 0.6 * conf + 0.25 * (edge_boost > 0) + 0.15 * (ready_boost > 0) + stance_boost * 0.0
        # Map
        if score >= 0.80:
            return "A"
        if score >= 0.65:
            return "B"
        if score >= 0.50:
            return "C"
        return "D"
    except Exception:
        return "C"
