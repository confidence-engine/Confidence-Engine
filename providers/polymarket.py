"""
Polymarket provider adapter
- Discover BTC/ETH prediction markets via public API
- Filter by resolution source, end date window, and liquidity

Note: Network calls are abstracted behind a fetch function for easy mocking in tests.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta, timezone
import json
import os
import urllib.request
import urllib.parse

def _append_query(url: str, extra: Dict[str, str]) -> str:
    try:
        parts = urllib.parse.urlparse(url)
        q = dict(urllib.parse.parse_qsl(parts.query))
        q.update({k: str(v) for k, v in extra.items()})
        new_query = urllib.parse.urlencode(q)
        return urllib.parse.urlunparse((parts.scheme, parts.netloc, parts.path, parts.params, new_query, parts.fragment))
    except Exception:
        # Fallback naive append
        sep = '&' if '?' in url else '?'
        return f"{url}{sep}" + "&".join(f"{k}={urllib.parse.quote(str(v))}" for k,v in extra.items())

# Default API endpoint (kept simple; tests will mock fetch)
DEFAULT_API_URL = os.getenv(
    "POLYMARKET_API_URL",
    "https://gamma-api.polymarket.com/markets?closed=false",
)


def _default_fetch(url: str) -> Dict[str, Any]:
    # Some Polymarket endpoints reject requests without a UA. Add simple headers.
    headers = {
        "User-Agent": os.getenv("TB_HTTP_USER_AGENT", "TracerBullet/1.0 (+https://github.com/confidence-engine/Tracer-Bullet)"),
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()
        return json.loads(data)


def _is_btc_eth_market(m: Dict[str, Any]) -> bool:
    # Allow configurable crypto assets via env, default to BTC/ETH
    assets_env = os.getenv("TB_POLYMARKET_ASSETS", "BTC,ETH")
    assets = [a.strip().upper() for a in assets_env.split(",") if a.strip()]
    keywords: List[str] = []
    for a in assets:
        if a == "BTC":
            keywords += ["bitcoin", "btc"]
        elif a == "ETH":
            keywords += ["ethereum", "eth"]
        elif a == "SOL":
            keywords += ["solana", "sol"]
        else:
            # generic fallback: also look for the ticker itself
            keywords.append(a.lower())
    title = (m.get("title") or m.get("question") or m.get("name") or m.get("slug") or "").lower()
    tags = m.get("tags") or []
    tags_text = " ".join([str(t).lower() for t in tags])
    haystack = f"{title} {tags_text}"
    return any(k in haystack for k in keywords)


def _has_clear_resolution(m: Dict[str, Any]) -> bool:
    # Allow bypass via env toggle
    if os.getenv("TB_POLYMARKET_REQUIRE_RESOLUTION", "1") == "0":
        return True
    res = (m.get("resolutionSource") or m.get("resolution_source") or "").strip()
    return bool(res)


def _parse_end_date(m: Dict[str, Any]) -> Optional[datetime]:
    # Try multiple common fields seen in Polymarket APIs
    dt = (
        m.get("endDate")
        or m.get("end_date")
        or m.get("closesAt")
        or m.get("endTime")
        or m.get("closeTime")
        or m.get("expiry")
    )
    if not dt:
        return None
    try:
        return datetime.fromisoformat(dt.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _liquidity(m: Dict[str, Any]) -> float:
    for k in ("liquidity", "liquidityUSD", "liquidity_usd", "volume24h", "volume", "totalVolumeUSD"):
        v = m.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def get_btc_eth_markets(
    *,
    min_liquidity: float = 1000.0,
    min_weeks: int = 1,
    max_weeks: int = 12,
    max_items: Optional[int] = None,
    fetch: Callable[[str], Dict[str, Any]] = _default_fetch,
    api_url: str = DEFAULT_API_URL,
) -> List[Dict[str, Any]]:
    """
    Fetch BTC/ETH markets and apply filters.
    Returns a list of market dicts (raw), trimmed by filters and limits.
    """
    # Allow increasing limit via env to capture more markets in a single call
    limit = os.getenv("TB_POLYMARKET_LIMIT")
    if limit and "limit=" not in api_url:
        api_url = _append_query(api_url, {"limit": str(limit)})
    try:
        payload = fetch(api_url)
    except Exception:
        return []

    items = payload if isinstance(payload, list) else payload.get("markets") or payload.get("data") or []
    debug = os.getenv("TB_POLYMARKET_DEBUG", "0") == "1"
    if not isinstance(items, list):
        return []

    now = datetime.now(timezone.utc)
    lo = now + timedelta(weeks=max(0, int(min_weeks)))
    hi = now + timedelta(weeks=max(0, int(max_weeks)))

    if debug:
        print(f"[Polymarket] fetched {len(items)} items from API")

    out: List[Dict[str, Any]] = []
    for m in items:
        if not _is_btc_eth_market(m):
            if debug:
                t = (m.get("title") or m.get("question") or m.get("name") or "").strip()
                print(f"[Polymarket] skip non-target asset: {t}")
            continue
        if not _has_clear_resolution(m):
            if debug:
                t = (m.get("title") or m.get("question") or m.get("name") or "").strip()
                print(f"[Polymarket] skip no resolution: {t}")
            continue
        ed = _parse_end_date(m)
        if os.getenv("TB_POLYMARKET_REQUIRE_ENDDATE", "1") == "1":
            if not ed or ed < lo or ed > hi:
                if debug:
                    t = (m.get("title") or m.get("question") or m.get("name") or "").strip()
                    print(f"[Polymarket] skip endDate window: {t}, end={ed}")
                continue
        if _liquidity(m) < float(min_liquidity):
            if debug:
                t = (m.get("title") or m.get("question") or m.get("name") or "").strip()
                print(f"[Polymarket] skip low liquidity: {t}, liq={_liquidity(m)}")
            continue
        out.append(m)

    # If nothing matched and search fallback is enabled, query by keywords
    if not out and os.getenv("TB_POLYMARKET_ENABLE_SEARCH", "1") == "1":
        # Build keywords from env assets
        assets_env = os.getenv("TB_POLYMARKET_ASSETS", "BTC,ETH")
        assets = [a.strip().upper() for a in assets_env.split(",") if a.strip()]
        kw: List[str] = []
        for a in assets:
            kw.extend({
                "BTC": ["bitcoin", "btc"],
                "ETH": ["ethereum", "eth"],
                "SOL": ["solana", "sol"],
            }.get(a, [a.lower()]))
        seen_ids = set()
        def _id(m):
            return m.get("conditionId") or m.get("id") or m.get("slug") or m.get("questionId")
        for token in kw:
            try:
                search_url = _append_query(api_url, {"search": token})
                payload2 = fetch(search_url)
            except Exception:
                continue
            items2 = payload2 if isinstance(payload2, list) else payload2.get("markets") or payload2.get("data") or []
            if debug:
                print(f"[Polymarket] search '{token}' returned {len(items2) if isinstance(items2, list) else 0} items")
            for m in items2 or []:
                mid = _id(m)
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                ed = _parse_end_date(m)
                if os.getenv("TB_POLYMARKET_REQUIRE_ENDDATE", "1") == "1":
                    if not ed:
                        continue
                    if ed < now + timedelta(weeks=max(0, int(min_weeks))) or ed > now + timedelta(weeks=max(0, int(max_weeks))):
                        continue
                if _liquidity(m) < float(min_liquidity):
                    continue
                if not _is_btc_eth_market(m):
                    continue
                out.append(m)

    # crude ranking by liquidity desc then soonest end
    out.sort(key=lambda x: (-_liquidity(x), _parse_end_date(x) or hi))

    if isinstance(max_items, int) and max_items > 0:
        out = out[:max_items]
    return out
