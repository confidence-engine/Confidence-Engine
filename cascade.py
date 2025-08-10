from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


_STOPWORDS = {
    "the",
    "a",
    "an",
    "for",
    "to",
    "of",
    "and",
    "in",
    "on",
    "with",
    "by",
    "at",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "this",
    "that",
    "these",
    "those",
}


_punct_re = re.compile(r"[^a-z0-9\s]")


def _normalize_text(text: str) -> str:
    t = (text or "").lower()
    t = _punct_re.sub(" ", t)
    tokens = [tok for tok in t.split() if tok and tok not in _STOPWORDS]
    return " ".join(tokens)


_ALIASES = {
    "btc": "bitcoin",
    "eth": "ethereum",
}

_UP_SYNS = {
    "soar",
    "soars",
    "soared",
    "surge",
    "surges",
    "surged",
    "skyrocket",
    "skyrockets",
    "skyrocketed",
    "rally",
    "rallies",
    "jump",
    "jumps",
    "spike",
    "spikes",
    "pump",
    "pumps",
    "rocket",
    "rockets",
    "moon",
    "moons",
    "gain",
    "gains",
    "gained",
    "rise",
    "rises",
    "rose",
    "breakout",
}

_DOWN_SYNS = {
    "fall",
    "falls",
    "fell",
    "drop",
    "drops",
    "dropped",
    "plunge",
    "plunges",
    "crash",
    "crashes",
    "dump",
    "dumps",
    "lose",
    "loses",
    "lost",
    "decline",
    "declines",
    "bearish",
}


def _canon(tok: str) -> str:
    tok = _ALIASES.get(tok, tok)
    if tok in _UP_SYNS:
        return "up"
    if tok in _DOWN_SYNS:
        return "down"
    return tok


def _token_set(text: str) -> set:
    tokens = _normalize_text(text).split()
    canon = [_canon(t) for t in tokens]
    return set(canon)


def _repetition_ratio(accepted: List[Dict], jaccard_threshold: float = 0.6) -> float:
    n = len(accepted or [])
    if n < 2:
        return 0.0
    token_sets = [_token_set(item.get("headline", "")) for item in accepted]
    clusters: List[set] = []  # store representative token sets
    duplicates = 0
    for ts in token_sets:
        matched = False
        for rep in clusters:
            union = len(ts | rep) or 1
            inter = len(ts & rep)
            j = inter / float(union)
            if j >= jaccard_threshold:
                duplicates += 1
                matched = True
                break
        if not matched:
            clusters.append(ts)
    return min(1.0, max(0.0, duplicates / float(n)))


def _extract_close_series(bars) -> List[float]:
    # Accepts pandas DataFrame/Series-like or simple iterables
    try:
        # pandas DataFrame with column 'close'
        s = getattr(bars, "close", None)
        if s is not None:
            try:
                # pandas Series
                return [float(s.iloc[i]) for i in range(len(s))]
            except Exception:
                try:
                    return [float(s[i]) for i in range(len(s))]
                except Exception:
                    pass
        # dict-like access
        if isinstance(bars, dict) and "close" in bars:
            seq = bars["close"]
            return [float(x) for x in seq]
    except Exception:
        pass
    # Fallback: treat as iterable of numbers
    try:
        return [float(x) for x in bars]
    except Exception:
        return []


def _extract_volume_series(bars) -> List[float]:
    try:
        s = getattr(bars, "volume", None)
        if s is not None:
            try:
                return [float(s.iloc[i]) for i in range(len(s))]
            except Exception:
                try:
                    return [float(s[i]) for i in range(len(s))]
                except Exception:
                    pass
        if isinstance(bars, dict) and "volume" in bars:
            seq = bars["volume"]
            return [float(x) for x in seq]
    except Exception:
        pass
    return []


def detect_cascade(accepted: List[Dict], bars) -> Dict:
    """
    Detect hype-only cascades and recommend reducing confidence.

    Returns dict with keys:
      - repetition_ratio: float in [0,1]
      - price_move_pct: abs percentage move over window
      - max_volume_z: max abs z-score of volume over window
      - tag: "HYPE_ONLY" or ""
      - confidence_delta: negative adjustment (e.g., -0.03) or 0.0
    """
    rr = _repetition_ratio(accepted)

    closes = _extract_close_series(bars)
    price_move_pct = 0.0
    if len(closes) >= 2 and closes[0] != 0:
        price_move_pct = abs((closes[-1] - closes[0]) / closes[0] * 100.0)

    vols = _extract_volume_series(bars)
    max_volume_z = 0.0
    # Require at least 3 points for a meaningful z-score estimate; else treat as negligible
    if len(vols) >= 3:
        mean = sum(vols) / float(len(vols))
        var = sum((v - mean) ** 2 for v in vols) / float(len(vols))
        std = var ** 0.5
        if std > 0:
            zs = [(v - mean) / std for v in vols]
            max_volume_z = max(abs(z) for z in zs)
        else:
            max_volume_z = 0.0

    tag = ""
    delta = 0.0
    if rr >= 0.5 and price_move_pct < 0.3 and max_volume_z < 0.3:
        tag = "HYPE_ONLY"
        delta = -0.03

    return {
        "repetition_ratio": round(float(rr), 3),
        "price_move_pct": round(float(price_move_pct), 3),
        "max_volume_z": round(float(max_volume_z), 3),
        "tag": tag,
        "confidence_delta": float(delta),
    }


