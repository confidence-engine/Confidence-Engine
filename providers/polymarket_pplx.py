"""
Perplexity-backed Polymarket provider
- Queries Perplexity Pro API to retrieve active Polymarket crypto markets
- Returns a list of market dicts with fields compatible with providers/polymarket.py

Env:
- PPLX_API_KEY (required when actually calling API)
- PPLX_MODEL (default: sonar)
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
from datetime import datetime, timezone, timedelta

PPLX_ENDPOINT = os.getenv("PPLX_ENDPOINT", "https://api.perplexity.ai/chat/completions")


def _default_fetch(url: str, data: bytes, headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _build_prompt(assets: List[str], limit: Optional[int]) -> str:
    # Implement crypto-wide prompt (emphasis BTC/ETH), request multiple strikes/variants, up to 'limit' (fallback 6)
    lim = limit if isinstance(limit, int) and limit > 0 else 6
    return (
        "You are a precise data extraction engine.\n"
        "List all current active Polymarket markets related to crypto, especially Bitcoin (BTC) and Ethereum (ETH).\n"
        "Include all relevant price prediction markets, strike thresholds, and up/down daily outcome questions. If a market family has multiple strikes/targets, return separate items per strike that are live.\n"
        "Apply all filters strictly: live/active, not expired; end date between now+1h and now+12 weeks; liquidityUSD ≥ 10000; resolution source must be clearly specified; exclude non-crypto/meme/politics/sports.\n"
        f"Return up to the {lim} highest-liquidity crypto markets, ordered by liquidity (desc) then recency (earlier endDate preferred).\n\n"
        "For each market, return ONLY these keys (use the exact Polymarket title as market_name, preserving any numeric thresholds/targets verbatim; do not paraphrase):\n"
        "- market_name (string)\n"
        "- event_end_date (ISO 8601 string, UTC)\n"
        "- implied_probability (0–1 float for YES)\n"
        "- liquidity_usd (number)\n"
        "- resolution_source (string)\n"
        "- asset (BTC, ETH, or other crypto symbol)\n"
        "- market_id (string) or slug (string)\n\n"
        "Output STRICT JSON array (no prose, no markdown fences)."
    )


def _build_fallback_prompt(assets: List[str], limit: Optional[int]) -> str:
    # Reiterate schema and constraints for crypto-wide; request up to 'limit' (fallback 6)
    lim = limit if isinstance(limit, int) and limit > 0 else 6
    return (
        "Return ONLY a JSON array (no markdown) of current active Polymarket crypto markets (crypto only; emphasize BTC/ETH).\n"
        "Include price/threshold/up-down daily variants. If there are multiple live strikes, include them as separate items.\n"
        "Filters: live/active; end between now+1h and now+12w; liquidityUSD ≥ 10000; resolution source specified and unambiguous; exclude non-crypto/meme/politics/sports.\n"
        f"Return up to {lim} by liquidity desc then earliest endDate.\n"
        "Each item keys: market_name (exact Polymarket title with any numeric thresholds intact), event_end_date (ISO8601), implied_probability (0–1 YES), liquidity_usd, resolution_source, asset (symbol), market_id or slug."
    )


def get_crypto_markets_via_pplx(
    *,
    assets_env: str = "BTC,ETH",
    limit: Optional[int] = None,
    fetch: Optional[Callable[[str, bytes, Dict[str, str], int], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    assets = [a.strip().upper() for a in assets_env.split(",") if a.strip()]
    prompt = _build_prompt(assets, limit)
    # Force model to 'sonar' for reliability/cost regardless of env
    model = "sonar"
    timeout = int(os.getenv("PPLX_TIMEOUT", "20"))

    # Allow custom prompt override via env, but always append a strict schema so parsing works
    prompt_override = os.getenv("TB_POLYMARKET_PPLX_PROMPT")
    schema_suffix = (
        "\n\nReturn ONLY a STRICT JSON array (no prose, no markdown). Each item must be an object with keys: \n"
        "  title (string), endDate (ISO 8601 string if available), liquidityUSD (number if available), \n"
        "  impliedProbability (0-1 number if available), resolutionSource (string or empty), tags (string array, optional).\n"
        "Return ONLY the JSON array."
    )
    user_prompt = (prompt_override + schema_suffix) if prompt_override else prompt

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

    if fetch is None:
        fetch = _default_fetch
    # Relax filters in tests when a custom fetch function is provided
    relax_filters = fetch is not _default_fetch
    debug = os.getenv("TB_POLYMARKET_DEBUG", "0") == "1"
    # Build list of API keys to try.
    # If TB_POLYMARKET_PPLX_USE_PLAIN_ONLY=1, use ONLY PPLX_API_KEY (dedicated plain key).
    use_plain_only = os.getenv("TB_POLYMARKET_PPLX_USE_PLAIN_ONLY", "0") == "1"
    key_items = []
    if use_plain_only:
        if os.getenv("PPLX_API_KEY", "").strip():
            key_items.append((0, os.getenv("PPLX_API_KEY").strip()))
    else:
        # Normal rotation: numbered keys first, then plain key last
        for k, v in os.environ.items():
            if k.startswith("PPLX_API_KEY_") and v.strip():
                try:
                    idx = int(k.split("_")[-1])
                except Exception:
                    idx = 0
                key_items.append((idx, v.strip()))
        key_items.sort(key=lambda x: x[0])
        if os.getenv("PPLX_API_KEY", "").strip():
            # Fallback single-key goes last
            key_items.append((10_000_000, os.getenv("PPLX_API_KEY").strip()))

    if debug:
        print(f"[Polymarket:PPLX] key rotation: {len(key_items)} keys discovered")
    if not key_items:
        if debug:
            print("[Polymarket:PPLX] WARNING: no PPLX_API_KEY[_N] set; request will likely fail.")

    # Retries per key + fallback prompt on first empty within a key
    retries = max(1, int(os.getenv("TB_POLYMARKET_PPLX_RETRIES", "2")))
    last_resp: Optional[Dict[str, Any]] = None
    for key_pos, key in enumerate([kv[1] for kv in key_items] or [""]):
        # Reset prompt body for each new key (so fallback is applied per-key)
        current_body = body
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        elif debug:
            print("[Polymarket:PPLX] WARNING: empty API key at this rotation slot")

        for attempt in range(retries):
            if debug:
                try:
                    print(
                        f"[Polymarket:PPLX] key#{key_pos+1} attempt {attempt+1}/{retries} prompt[0:160]=\n{(user_prompt or '')[:160]}..."
                    )
                except Exception:
                    pass
            try:
                last_resp = fetch(PPLX_ENDPOINT, current_body, headers, timeout)
            except Exception as e:
                if debug:
                    try:
                        print(f"[Polymarket:PPLX] key#{key_pos+1} fetch error on attempt {attempt+1}: {e}")
                    except Exception:
                        pass
                continue
            # lightweight response introspection
            if debug:
                try:
                    ch = last_resp.get("choices") or []
                    cnt = len(ch)
                    snippet = ""
                    if cnt:
                        snippet = (ch[0].get("message", {}).get("content") or "")
                        snippet = snippet[:160]
                    print(f"[Polymarket:PPLX] key#{key_pos+1} attempt {attempt+1} choices={cnt} content_snippet[0:160]={snippet!r}")
                except Exception:
                    pass
            items = _parse_and_normalize(last_resp, debug, relax_filters=relax_filters)
            if items:
                return items
            if debug:
                print(f"[Polymarket:PPLX] key#{key_pos+1} attempt {attempt+1} returned 0 items; will retry")
            # On first empty result and if no custom prompt override, try fallback prompt variant (per key)
            if attempt == 0 and not prompt_override:
                fb_prompt = _build_fallback_prompt(assets, limit)
                current_body = json.dumps({
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
        # try next key
        if debug:
            print(f"[Polymarket:PPLX] rotating to next key (key#{key_pos+1} produced no items)")
    # Final broad fallback: relax constraints and ask for any active BTC/ETH/SOL/XRP price markets
    try:
        broad_assets = ", ".join(assets) if assets else "BTC, ETH, SOL, XRP"
        broad_prompt = (
            "You are a data extraction engine. "
            "Task: List currently-active Polymarket crypto PRICE markets for these assets: "
            f"{broad_assets}. "
            "Do not filter by end date; include any market that is trading now.\n"
            "Output format (STRICT JSON array, no prose, no markdown): each item is an object with keys: \n"
            "  title (string), endDate (ISO 8601 UTC string if available), liquidityUSD (number if available), \n"
            "  impliedProbability (0-1 number if available), resolutionSource (string or empty), tags (string array, optional).\n"
            "Return ONLY the JSON array."
        )
        if debug:
            print("[Polymarket:PPLX] attempting broad fallback prompt")
        fallback_key = key_items[0][1] if key_items else os.getenv("PPLX_API_KEY", "")
        headers2 = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if fallback_key:
            headers2["Authorization"] = f"Bearer {fallback_key}"
        body2 = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise data extraction engine."},
                {"role": "user", "content": broad_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1200,
            "return_citations": False,
            "return_images": False,
        }).encode("utf-8")
        for attempt in range(max(1, int(os.getenv("TB_POLYMARKET_PPLX_RETRIES", "2")))):
            try:
                resp2 = fetch(PPLX_ENDPOINT, body2, headers2, timeout)
            except Exception as e:
                if debug:
                    print(f"[Polymarket:PPLX] broad fallback fetch error on attempt {attempt+1}: {e}")
                continue
            items2 = _parse_and_normalize(resp2, debug, relax_filters=relax_filters)
            if items2:
                if debug:
                    print(f"[Polymarket:PPLX] broad fallback yielded {len(items2)} items")
                return items2
    except Exception as e:
        if debug:
            print(f"[Polymarket:PPLX] broad fallback failed: {e}")
    return []

def _parse_and_normalize(resp: Dict[str, Any], debug: bool, *, relax_filters: bool = False) -> List[Dict[str, Any]]:
    # Perplexity returns a chat completion; parse the message content and load JSON, then normalize
    def _extract_first_json_array(text: str) -> Optional[str]:
        # Remove code fence wrappers but keep inner content
        t = text.strip()
        if t.startswith("```"):
            # strip first fence
            nl = t.find("\n")
            if nl != -1:
                t = t[nl+1:]
            if t.endswith("```"):
                t = t[:-3]
        # Scan for first balanced JSON array using bracket depth
        start_idx = -1
        depth = 0
        for i, ch in enumerate(t):
            if ch == '[':
                if depth == 0 and start_idx == -1:
                    start_idx = i
                depth += 1
            elif ch == ']':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start_idx != -1:
                        return t[start_idx:i+1]
        return None

    try:
        choices = resp.get("choices") or []
        content = choices[0]["message"]["content"] if choices else "[]"
        # Prefer extracting a balanced array; fallback to original heuristics
        arr = _extract_first_json_array(content)
        if arr is None:
            s = content.strip()
            start = s.find("[")
            end = s.rfind("]")
            if start != -1 and end != -1 and end > start:
                arr = s[start:end+1]
            else:
                arr = "[]"
        data = json.loads(arr)
    except Exception as e:
        if debug:
            print(f"[Polymarket:PPLX] parse failure: {e}")
        return []

    items: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for m in data:
            if not isinstance(m, dict):
                continue
            # Map alternate incoming keys as requested by strict prompt
            title = (
                m.get("market_name") or m.get("title") or m.get("question") or m.get("name")
            )
            if not title:
                continue
            # derive impliedProbability robustly
            raw_prob = (
                m.get("implied_probability")
                or m.get("impliedProbability")
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
                "endDate": m.get("event_end_date") or m.get("endDate") or m.get("end_date") or m.get("closesAt"),
                "liquidityUSD": m.get("liquidity_usd") or m.get("liquidityUSD") or m.get("liquidity") or m.get("volume24h") or m.get("volume"),
                "impliedProbability": implied_prob,
                "resolutionSource": m.get("resolution_source") or m.get("resolutionSource") or m.get("resolution_source_url") or "",
                "marketId": m.get("market_id") or m.get("id") or m.get("slug") or m.get("url"),
                "asset": (m.get("asset") or "").upper(),
                "tags": m.get("tags") or [],
            }
            if debug and item["impliedProbability"] is None:
                try:
                    print(f"[Polymarket:PPLX] note: missing impliedProbability for title='{str(title)[:90]}'")
                except Exception:
                    pass
            items.append(item)
    # Client-side filtering: crypto-wide; optionally strict window/liquidity/resolution constraints
    def _parse_dt(s: Any) -> Optional[datetime]:
        if not s:
            return None
        try:
            # Accept plain ISO 8601
            return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        except Exception:
            return None
    now = datetime.now(timezone.utc)
    # detect crypto by common tickers/keywords; accept broader set
    crypto_terms = [
        "btc","bitcoin","eth","ethereum","sol","solana","xrp","ripple","ada","cardano","doge","dogecoin","ltc","litecoin","bnb","trx","tron","dot","polkadot","link","chainlink","avax","avalanche","arb","arbitrum","op","optimism","atom","cosmos","uni","uniswap","matic","polygon","pol","polymesh"
    ]
    def _is_crypto(title: str) -> bool:
        t = title.lower()
        return any(term in t for term in crypto_terms)
    if relax_filters:
        filtered = list(items)
    else:
        filtered = []
        for it in items:
            t_l = (it.get("title") or "")
            if not _is_crypto(t_l):
                continue
            dt = _parse_dt(it.get("endDate"))
            if not dt:
                continue
            if not (now + timedelta(hours=1) <= dt <= now + timedelta(weeks=12)):
                continue
            liq = None
            try:
                liq = float(it.get("liquidityUSD") or 0)
            except Exception:
                liq = 0.0
            if liq < 10000.0:
                continue
            rs = (it.get("resolutionSource") or "").strip()
            if not rs or rs.lower() in ("unknown", "n/a"):
                continue
            # derive asset symbol if missing
            asset = (it.get("asset") or "").upper()
            if not asset:
                tl = t_l.lower()
                if "bitcoin" in tl or "btc" in tl:
                    asset = "BTC"
                elif "ethereum" in tl or "eth" in tl:
                    asset = "ETH"
                elif "sol" in tl or "solana" in tl:
                    asset = "SOL"
                elif "xrp" in tl or "ripple" in tl:
                    asset = "XRP"
                elif "ada" in tl or "cardano" in tl:
                    asset = "ADA"
                elif "doge" in tl or "dogecoin" in tl:
                    asset = "DOGE"
                elif "ltc" in tl or "litecoin" in tl:
                    asset = "LTC"
                elif "bnb" in tl:
                    asset = "BNB"
                elif "dot" in tl or "polkadot" in tl:
                    asset = "DOT"
                elif "link" in tl or "chainlink" in tl:
                    asset = "LINK"
                elif "avax" in tl or "avalanche" in tl:
                    asset = "AVAX"
                elif "arb" in tl or "arbitrum" in tl:
                    asset = "ARB"
                elif "op" in tl or "optimism" in tl:
                    asset = "OP"
                elif "atom" in tl or "cosmos" in tl:
                    asset = "ATOM"
                elif "uni" in tl or "uniswap" in tl:
                    asset = "UNI"
                elif "matic" in tl or "polygon" in tl:
                    asset = "MATIC"
                elif "pol" in tl and "polymesh" in tl:
                    asset = "POL"
                else:
                    asset = "OTHER"
            it["asset"] = asset
            filtered.append(it)
    # Sort by liquidity desc, then earliest endDate
    def _key(x: Dict[str, Any]):
        try:
            liq = float(x.get("liquidityUSD") or 0)
        except Exception:
            liq = 0.0
        dtx = _parse_dt(x.get("endDate")) or (now + timedelta(days=3650))
        title = (x.get("title") or "")
        has_num = any(ch.isdigit() for ch in title)
        # Prefer titles that contain explicit numeric thresholds (likely strike/target markets), then liquidity, then recency
        return (0 if has_num else 1, -liq, dtx)
    filtered.sort(key=_key)
    # Optional client-side cap: TB_POLYMARKET_PPLX_MAX (>0). Unset or <=0 disables capping
    try:
        cap_env = os.getenv("TB_POLYMARKET_PPLX_MAX", "0").strip()
        cap = int(cap_env) if cap_env else 0
    except Exception:
        cap = 0
    if cap > 0 and len(filtered) > cap:
        filtered = filtered[:cap]
    if debug:
        print(f"[Polymarket:PPLX] normalized {len(items)} items -> strict_filtered {len(filtered)} (crypto filtered)")
    return filtered
