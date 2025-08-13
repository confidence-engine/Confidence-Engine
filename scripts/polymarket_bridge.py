"""
Polymarket bridge:
- Fetch BTC/ETH markets via providers.polymarket
- Filter/map to internal stance/readiness/edge labels
- Generate number-free rationale for TG/Discord; artifacts keep numeric fields
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timezone
import os

try:
    from providers.polymarket import get_btc_eth_markets
except Exception:
    from .providers.polymarket import get_btc_eth_markets  # type: ignore

# Optional Perplexity-backed adapter
def _get_markets_via_source(
    *,
    source: str,
    min_liquidity: float,
    min_weeks: int,
    max_weeks: int,
    max_items: int,
    fetch_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    src = (source or "native").lower()
    if src == "pplx":
        try:
            try:
                from providers.polymarket_pplx import get_crypto_markets_via_pplx
            except Exception:
                from .providers.polymarket_pplx import get_crypto_markets_via_pplx  # type: ignore
            assets = os.getenv("TB_POLYMARKET_ASSETS", "BTC,ETH")
            # Prefer explicit TB_POLYMARKET_LIMIT; otherwise use TB_POLYMARKET_MAX_ITEMS as request size
            limit_env = os.getenv("TB_POLYMARKET_LIMIT")
            max_items_env = os.getenv("TB_POLYMARKET_MAX_ITEMS")
            limit: Optional[int] = None
            if limit_env and limit_env.isdigit():
                limit = int(limit_env)
            elif max_items_env and max_items_env.isdigit():
                limit = int(max_items_env)
            items = get_crypto_markets_via_pplx(assets_env=assets, limit=limit, fetch=None)
            # Optional title keyword filter to bias to BTC/ETH/SOL
            kw_env = os.getenv("TB_POLYMARKET_TITLE_KEYWORDS", "BTC,Bitcoin,ETH,Ethereum,SOL,Solana")
            kws = [k.strip().lower() for k in kw_env.split(",") if k.strip()]
            if kws:
                def _ok(t: str) -> bool:
                    lt = (t or "").lower()
                    return any(k in lt for k in kws)
                before = len(items)
                items = [m for m in items if _ok(str(m.get("title") or m.get("question") or m.get("name") or ""))]
                if os.getenv("TB_POLYMARKET_DEBUG", "0") == "1":
                    print(f"[Polymarket] keyword filter: {before} -> {len(items)} using {kws}")
            # Soft cap to desired max to avoid over-long sections
            try:
                desired_max = int(max_items_env) if (max_items_env and max_items_env.isdigit()) else None
            except Exception:
                desired_max = None
            if desired_max and len(items) > desired_max:
                items = items[:desired_max]
                if os.getenv("TB_POLYMARKET_DEBUG", "0") == "1":
                    print(f"[Polymarket] capped to {desired_max} items after filter")
            if not items and os.getenv("TB_POLYMARKET_FALLBACK_NATIVE", "0") == "1":
                # Try native as a last resort
                return get_btc_eth_markets(
                    min_liquidity=min_liquidity,
                    min_weeks=min_weeks,
                    max_weeks=max_weeks,
                    max_items=max_items,
                    fetch=fetch_fn if fetch_fn else None,  # type: ignore
                )
            return items
        except Exception:
            return []
    # default: native provider
    return get_btc_eth_markets(
        min_liquidity=min_liquidity,
        min_weeks=min_weeks,
        max_weeks=max_weeks,
        max_items=max_items,
        fetch=fetch_fn if fetch_fn else None,  # type: ignore
    )

try:
    from .evidence_lines import strip_numbers_for_chat
except Exception:
    from scripts.evidence_lines import strip_numbers_for_chat  # type: ignore


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _time_to_end_weeks(end_dt: Optional[datetime]) -> float:
    if not end_dt:
        return 99.0
    delta = end_dt - _now_utc()
    return max(0.0, delta.total_seconds() / (7 * 24 * 3600.0))


def _readiness_from_weeks(w: float) -> str:
    if w <= 2:
        return "Now"
    if w <= 6:
        return "Near"
    return "Later"


def _edge_label(internal_prob: Optional[float], implied_prob: Optional[float], tol: Optional[float] = None) -> str:
    try:
        # Allow env override; default slightly tighter at 0.02
        if tol is None:
            try:
                tol = float(os.getenv("TB_POLY_EDGE_TOL", "0.02"))
            except Exception:
                tol = 0.02
        if internal_prob is None or implied_prob is None:
            return "in-line"
        diff = float(internal_prob) - float(implied_prob)
        if diff > float(tol):
            return "market cheap"
        if diff < -float(tol):
            return "market rich"
        return "in-line"
    except Exception:
        return "in-line"


def _stance(edge: str, readiness: str) -> str:
    if edge in ("market cheap", "market rich") and readiness in ("Now", "Near"):
        return "Engage"
    if edge in ("market cheap", "market rich"):
        return "Stalk"
    return "Stand Aside"


def _rationale(edge: str, readiness: str, title: str) -> str:
    parts: List[str] = []
    # sentence 1
    if edge == "market cheap":
        parts.append("Market pricing looks favorable versus our view.")
    elif edge == "market rich":
        parts.append("Market pricing looks stretched versus our view.")
    else:
        parts.append("Market pricing is broadly in-line with our view.")
    # sentence 2
    if readiness == "Now":
        parts.append("Timing is actionable; manage risk and execution.")
    elif readiness == "Near":
        parts.append("Timing is approaching; monitor for clean triggers.")
    else:
        parts.append("Timing is later; keep on watch and reassess.")
    text = " ".join(parts)
    return strip_numbers_for_chat(text)


def _estimate_internal_prob(title: str, implied_prob: Optional[float], context: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """
    Simple, deterministic estimator for internal probability based on title keywords and env bias.

    Controls:
    - TB_POLYMARKET_INTERNAL_ENABLE=1 to enable
    - TB_POLYMARKET_INTERNAL_MODE=bias (default) [future: extend]
    - TB_POLYMARKET_INTERNAL_BIAS=0.05 (added or subtracted based on keywords)
    - TB_POLYMARKET_INTERNAL_ACTION_BIAS=0.08 (bias from BTC/ETH action signal)
    Calibration (uses asset signals when provided via context):
    - TB_POLY_INT_ALIGN_W=0.2   (0-1)
    - TB_POLY_INT_READY_W=0.2   (0-1)
    - TB_POLY_INT_ACTION_W=0.4  (0-1)
    - TB_POLY_INT_RISK_W=0.2    (0-1)
    - TB_POLY_INT_MAX_SHIFT=0.2 (cap absolute adjustment)
    - Keywords YES: above, up, reach, at least, over
      Keywords NO: below, down, under, less than
    """
    if os.getenv("TB_POLYMARKET_INTERNAL_ENABLE", "0") != "1":
        return None
    try:
        mode = os.getenv("TB_POLYMARKET_INTERNAL_MODE", "bias").lower()
        bias = float(os.getenv("TB_POLYMARKET_INTERNAL_BIAS", "0.05"))
        act_bias = float(os.getenv("TB_POLYMARKET_INTERNAL_ACTION_BIAS", "0.08"))
        w_align = float(os.getenv("TB_POLY_INT_ALIGN_W", "0.2"))
        w_ready = float(os.getenv("TB_POLY_INT_READY_W", "0.2"))
        w_action = float(os.getenv("TB_POLY_INT_ACTION_W", "0.4"))
        w_risk = float(os.getenv("TB_POLY_INT_RISK_W", "0.2"))
        max_shift = float(os.getenv("TB_POLY_INT_MAX_SHIFT", "0.2"))
    except Exception:
        mode = "bias"
        bias = 0.05
        act_bias = 0.08
        w_align = 0.2
        w_ready = 0.2
        w_action = 0.4
        w_risk = 0.2
        max_shift = 0.2
    if implied_prob is None:
        return None
    p = float(implied_prob)
    if mode == "bias":
        t = (title or "").lower()
        yes_kw = ("above", "up", "reach", "at least", "over")
        no_kw = ("below", "down", "under", "less than")
        dirn = 0
        if any(k in t for k in yes_kw):
            dirn = +1
        if any(k in t for k in no_kw):
            dirn = -1 if dirn == 0 else dirn  # preserve YES hit if both
        # Action bias from context (if BTC/ETH signals provided)
        act_dirn = 0
        try:
            asset_key = None
            if ("btc" in t) or ("bitcoin" in t):
                asset_key = "BTC"
            elif ("eth" in t) or ("ethereum" in t):
                asset_key = "ETH"
            if context and asset_key:
                info = context.get(asset_key) or context.get(f"{asset_key}/USD") or {}
                action = str((info.get("action") or info.get("th_action") or "")).lower()
                if action in ("buy", "long"):
                    act_dirn = +1
                elif action in ("sell", "short"):
                    act_dirn = -1
        except Exception:
            pass
        # Weighted calibration from signals
        calib_shift = 0.0
        debug_info: Dict[str, Any] = {}
        try:
            if context and asset_key:
                info = context.get(asset_key) or {}
                # readiness score
                r = str((info.get("readiness") or "")).lower()
                r_score = 1.0 if r == "now" else (0.6 if r == "near" else 0.2 if r == "later" else 0.0)
                # alignment score [0..1]
                a_score = float(info.get("align_score") or 0.0)
                if a_score > 1.0:
                    a_score = 1.0
                if a_score < 0.0:
                    a_score = 0.0
                # risk score as confidence proxy
                rb = str((info.get("risk_band") or "")).lower()
                risk_score = 1.0 if rb == "high" else (0.6 if rb == "medium" else 0.3 if rb == "low" else 0.5)
                # action sign already derived: act_dirn in {-1,0,1}
                action_score = 1.0 if act_dirn == +1 else (-1.0 if act_dirn == -1 else 0.0)
                # combine
                # Directional preference from title keywords: +1 for up/above, -1 for down/below, 0 if unknown
                dir_pref = +1 if any(k in t for k in yes_kw) else (-1 if any(k in t for k in no_kw) else 0)
                # composite in [-1,1]
                composite = (
                    w_action * action_score +
                    w_ready * (2*r_score - 1) +
                    w_align * (2*a_score - 1) +
                    w_risk * (2*risk_score - 1)
                )
                # apply toward YES if dir_pref>=0; toward NO if dir_pref<0
                dir_mult = 1.0 if dir_pref >= 0 else -1.0
                calib_shift = dir_mult * max_shift * composite
                # collect debug components
                debug_info = {
                    "asset": asset_key,
                    "r": r,
                    "r_score": round(r_score, 3),
                    "a_score": round(a_score, 3),
                    "risk_band": rb,
                    "risk_score": round(risk_score, 3),
                    "action_score": round(action_score, 3),
                    "dir_pref": dir_pref,
                    "composite": round(composite, 3),
                    "dir_mult": dir_mult,
                    "max_shift": round(max_shift, 3),
                }
        except Exception:
            calib_shift = 0.0
        p2 = p + (bias * dirn) + (act_bias * act_dirn) + calib_shift
        # clamp to [0,1]
        if p2 < 0.0:
            p2 = 0.0
        if p2 > 1.0:
            p2 = 1.0
        # optional debug print
        try:
            if os.getenv("TB_POLYMARKET_DEBUG", "0") == "1":
                print(
                    f"[Polymarket][internal] title='{(title or '').strip()[:90]}' implied={p:.3f} internal={p2:.3f} "
                    f"dir_kw={dirn:+d} act={act_dirn:+d} shift={calib_shift:+.3f} comps={debug_info}"
                )
        except Exception:
            pass
        return p2
    # fallback (no change)
    return None


def discover_and_map(
    *,
    min_liquidity: float,
    min_weeks: int,
    max_weeks: int,
    max_items: int,
    min_quality: float,
    fetch_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns list of mapped dicts with chat and artifact fields.
    Each item contains:
      - symbol: 'POLY:BTC' or 'POLY:ETH' (best-effort)
      - title
      - readiness, stance, edge_label
      - rationale_chat (number-free)
      - artifact fields: implied_prob, internal_prob, end_date_iso
    """
    markets = _get_markets_via_source(
        source=os.getenv("TB_POLYMARKET_SOURCE", "native"),
        min_liquidity=min_liquidity,
        min_weeks=min_weeks,
        max_weeks=max_weeks,
        max_items=max_items,
        fetch_fn=fetch_fn,
    )
    if os.getenv("TB_POLYMARKET_DEBUG", "0") == "1":
        print(f"[Polymarket] source={os.getenv('TB_POLYMARKET_SOURCE','native')} raw_items={len(markets)}")
    out: List[Dict[str, Any]] = []
    for m in markets:
        # optional quality gate
        q = m.get("quality") or m.get("score") or 1.0
        try:
            if float(q) < float(min_quality):
                continue
        except Exception:
            pass
        title = str(m.get("title") or m.get("question") or "").strip()
        lower_title = title.lower()
        symbol = "POLY:BTC" if ("btc" in lower_title or "bitcoin" in lower_title) else (
                 "POLY:ETH" if ("eth" in lower_title or "ethereum" in lower_title) else "POLY:CRYPTO")
        end_dt = None
        for k in ("endDate", "end_date", "closesAt"):
            val = m.get(k)
            if val:
                try:
                    from datetime import datetime, timezone
                    end_dt = datetime.fromisoformat(str(val).replace("Z", "+00:00")).astimezone(timezone.utc)
                    break
                except Exception:
                    pass
        weeks = _time_to_end_weeks(end_dt)
        readiness = _readiness_from_weeks(weeks)
        implied_prob = m.get("impliedProbability") or m.get("implied_prob") or m.get("yesPrice")
        if implied_prob is None and os.getenv("TB_POLYMARKET_DEBUG", "0") == "1":
            try:
                print(f"[Polymarket][internal] missing implied_prob for title='{title[:90]}'")
            except Exception:
                pass
        # internal TB prob: optional estimator; else default to implied
        est = _estimate_internal_prob(title, implied_prob, context=context)
        internal_prob = est if (est is not None) else (m.get("internal_prob") if m.get("internal_prob") is not None else implied_prob)
        edge = _edge_label(internal_prob, implied_prob)
        stance = _stance(edge, readiness)
        rationale = _rationale(edge, readiness, title)
        # Outcome lean based on implied probability (Yes vs No)
        implied_side = None
        implied_pct = None
        try:
            if implied_prob is not None:
                p = float(implied_prob)
                implied_side = "YES" if p >= 0.5 else "NO"
                implied_pct = round(p * 100.0)
        except Exception:
            pass
        if implied_side is None:
            implied_side = "UNKNOWN"
        # Outcome label (human)
        if implied_side == "YES":
            outcome_label = "Lean YES"
        elif implied_side == "NO":
            outcome_label = "Lean NO"
        else:
            outcome_label = "Split"
        out.append({
            "symbol": symbol,
            "title": title,
            "readiness": readiness,
            "stance": stance,
            "edge_label": edge,
            "rationale_chat": rationale,
            # derived outcome fields
            "implied_side": implied_side,
            "outcome_label": outcome_label,
            # artifact-only numeric fields
            "implied_prob": implied_prob,
            "internal_prob": internal_prob,
            "implied_pct": implied_pct,
            "end_date_iso": end_dt.isoformat() if end_dt else None,
        })
    return out


def discover_from_env(fetch_fn: Optional[Callable[[str], Dict[str, Any]]] = None, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if os.getenv("TB_ENABLE_POLYMARKET", "0") != "1":
        return []
    # Properly read liquidity and quality thresholds with independent defaults
    min_liq = float(os.getenv("TB_POLYMARKET_MIN_LIQUIDITY", "1000"))
    max_items = int(os.getenv("TB_POLYMARKET_MAX_ITEMS", "2"))
    min_quality = float(os.getenv("TB_POLYMARKET_MIN_QUALITY", "0.0"))
    min_weeks = int(os.getenv("TB_POLYMARKET_MIN_WEEKS", "1"))
    max_weeks = int(os.getenv("TB_POLYMARKET_MAX_WEEKS", "12"))
    mapped = discover_and_map(
        min_liquidity=min_liq,
        min_weeks=min_weeks,
        max_weeks=max_weeks,
        max_items=max_items,
        min_quality=min_quality,
        fetch_fn=fetch_fn,
        context=context,
    )
    if os.getenv("TB_POLYMARKET_DEBUG", "0") == "1":
        print(f"[Polymarket] mapped_items={len(mapped)} (source={os.getenv('TB_POLYMARKET_SOURCE','native')})")
    return mapped
