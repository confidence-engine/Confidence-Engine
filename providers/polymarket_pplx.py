"""
Perplexity-backed Polymarket provider
- Queries Perplexity Pro API to retrieve active Polymarket crypto markets
- Returns a list of market dicts with fields compatible with providers/polymarket.py

Env:
- PPLX_API_KEY (required when actually calling API)
- PPLX_MODEL (default: sonar-pro)
- PPLX_TIMEOUT (default: 20)
- TB_POLYMARKET_ASSETS (e.g., "BTC,ETH,SOL")
- TB_POLYMARKET_LIMIT (optional)

Testing:
- get_crypto_markets_via_pplx accepts a fetch callable for mocking.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
import json
import os
import urllib.request

PPLX_ENDPOINT = os.getenv("PPLX_ENDPOINT", "https://api.perplexity.ai/chat/completions")


def _default_fetch(url: str, data: bytes, headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _build_prompt(assets: List[str], limit: Optional[int]) -> str:
    assets_str = ", ".join(assets) if assets else "BTC, ETH, SOL"
    limit_str = f"Return at most {limit} items." if limit else "Return up to 10 items."
    return (
        "You are a data extraction engine. "
        "Task: List live, currently-active Polymarket crypto PRICE markets for these assets ONLY: "
        f"{assets_str}. {limit_str} "
        "Important requirements: \n"
        "- Markets must be on Polymarket (Gamma API/website) and be crypto price questions (not politics/sports).\n"
        "- Prefer intraday/daily markets like 'BTC up or down today' or 'ETH above X on <date>'.\n"
        "- Include only markets that are trading now (active/live).\n"
        "Output format (STRICT JSON array, no prose, no markdown): each item is an object with keys: \n"
        "  title (string), endDate (ISO 8601 UTC string if available), liquidityUSD (number if available), \n"
        "  impliedProbability (0-1 number if available), resolutionSource (string or empty), tags (string array, optional).\n"
        "Return ONLY the JSON array."
    )


def _build_fallback_prompt(assets: List[str], limit: Optional[int]) -> str:
    # Stricter, with explicit examples and schema reiteration
    assets_str = ", ".join(assets) if assets else "BTC, ETH, SOL"
    limit_str = f"Return at most {limit} items." if limit else "Return up to 10 items."
    return (
        "Return ONLY a JSON array (no markdown) of live Polymarket crypto PRICE markets. "
        f"Assets: {assets_str}. {limit_str} "
        "Examples of valid titles: 'Bitcoin Up or Down on August 13', 'Ethereum above 4200 on August 13?'. "
        "Each item must be an object with keys: title, endDate, liquidityUSD, impliedProbability, resolutionSource, tags. "
        "If a field is unknown, omit it. Include only markets currently trading (active)."
    )


def get_crypto_markets_via_pplx(
    *,
    assets_env: str = "BTC,ETH",
    limit: Optional[int] = None,
    fetch: Optional[Callable[[str, bytes, Dict[str, str], int], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    assets = [a.strip().upper() for a in assets_env.split(",") if a.strip()]
    prompt = _build_prompt(assets, limit)
    model = os.getenv("PPLX_MODEL", "sonar-pro")
    api_key = os.getenv("PPLX_API_KEY", "")
    timeout = int(os.getenv("PPLX_TIMEOUT", "20"))

    # Allow custom prompt override via env
    prompt_override = os.getenv("TB_POLYMARKET_PPLX_PROMPT")
    user_prompt = prompt_override if prompt_override else prompt

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise data extraction engine."},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1200,
        "return_citations": False,
        "return_images": False,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    _fetch = fetch or _default_fetch
    debug = os.getenv("TB_POLYMARKET_DEBUG", "0") == "1"
    if not api_key and debug:
        print("[Polymarket:PPLX] WARNING: PPLX_API_KEY not set; request may fail.")

    # Retries + fallback prompt on empty
    retries = max(1, int(os.getenv("TB_POLYMARKET_PPLX_RETRIES", "2")))
    last_resp: Optional[Dict[str, Any]] = None
    for attempt in range(retries):
        try:
            last_resp = _fetch(PPLX_ENDPOINT, body, headers, timeout)
        except Exception as e:
            if debug:
                print(f"[Polymarket:PPLX] fetch error on attempt {attempt+1}: {e}")
            continue
        # Try to parse; if items == 0 we can switch to fallback prompt once
        items = _parse_and_normalize(last_resp, debug)
        if items:
            return items
        if debug:
            print(f"[Polymarket:PPLX] attempt {attempt+1} returned 0 items; will retry")
        # On first empty result and if no custom prompt override, try fallback prompt variant
        if attempt == 0 and not prompt_override:
            fb_prompt = _build_fallback_prompt(assets, limit)
            body = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a precise data extraction engine."},
                    {"role": "user", "content": fb_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 1200,
                "return_citations": False,
                "return_images": False,
            }).encode("utf-8")
    return []

def _parse_and_normalize(resp: Dict[str, Any], debug: bool) -> List[Dict[str, Any]]:
    # Perplexity returns a chat completion; parse the message content and load JSON, then normalize
    try:
        choices = resp.get("choices") or []
        content = choices[0]["message"]["content"] if choices else "[]"
        s = content.strip()
        if s.startswith("```"):
            first_nl = s.find("\n")
            if first_nl != -1:
                s = s[first_nl + 1 : ]
            if s.endswith("```"):
                s = s[: -3]
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            s = s[start:end+1]
        data = json.loads(s)
    except Exception as e:
        if debug:
            print(f"[Polymarket:PPLX] parse failure: {e}")
        return []

    items: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for m in data:
            if not isinstance(m, dict):
                continue
            title = m.get("title") or m.get("question") or m.get("name")
            if not title:
                continue
            # derive impliedProbability robustly
            raw_prob = (
                m.get("impliedProbability")
                or m.get("implied_prob")
                or m.get("yesPrice")
                or m.get("p_yes")
                or m.get("probability")
                or m.get("price")
            )
            implied_prob = None
            try:
                if raw_prob is not None:
                    p = float(raw_prob)
                    if p > 1.0 and p <= 100.0:
                        p = p / 100.0
                    if 0.0 <= p <= 1.0:
                        implied_prob = p
            except Exception:
                implied_prob = None
            # Heuristic: binary phrasing without explicit price -> neutral 0.5 to allow internal estimation
            if implied_prob is None:
                lt = str(title).lower()
                if ("up or down" in lt) or ("above or below" in lt):
                    implied_prob = 0.5
            item = {
                "title": str(title),
                "endDate": m.get("endDate") or m.get("end_date") or m.get("closesAt"),
                "liquidityUSD": m.get("liquidityUSD") or m.get("liquidity") or m.get("volume24h") or m.get("volume"),
                "impliedProbability": implied_prob,
                "resolutionSource": m.get("resolutionSource") or m.get("resolution_source") or "",
                "tags": m.get("tags") or [],
            }
            if debug and item["impliedProbability"] is None:
                try:
                    print(f"[Polymarket:PPLX] note: missing impliedProbability for title='{str(title)[:90]}'")
                except Exception:
                    pass
            items.append(item)
    if debug:
        print(f"[Polymarket:PPLX] normalized {len(items)} items")
    return items
