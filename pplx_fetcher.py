import os
import json
import time
from typing import List, Tuple, Any, Optional
import httpx

PPLX_API_URL = "https://api.perplexity.ai/chat/completions"

DEFAULT_PROMPT = (
    "Return 12 concise crypto market headlines focused on Bitcoin (BTC). "
    "Prefer titles that explicitly mention Bitcoin/BTC. Avoid duplicates and clickbait. "
    "Respond ONLY as a JSON array of objects with keys: title, source, url. No preamble. No extra text."
)

def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

def _build_payload(prompt: str, hours: int = 24) -> dict:
    # Enforce recent results via web_search_options and a strict system instruction
    # to return ONLY a JSON array with no code fences or extra text.
    system_msg = (
        "You are a data extraction API. Respond ONLY with a valid JSON array. "
        "Do not include code fences, explanations, or any text outside the JSON array."
    )
    return {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"{prompt} Time window: last {hours} hours."}
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "web_search_options": {
            "search_recency_filter": "day",
            "enable_citation": True
        }
    }

def _attempt_request(
    client: httpx.Client,
    api_key: str,
    prompt: str,
    hours: int,
) -> Tuple[List[str], List[dict], Optional[str]]:
    payload = _build_payload(prompt, hours=hours)
    try:
        r = client.post(PPLX_API_URL, headers=_headers(api_key), json=payload)
        if r.status_code != 200:
            return [], [], f"HTTP {r.status_code} {r.text[:200]}"
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            text = str(content or "")
            # Strip Markdown code fences if present
            if text.strip().startswith("```"):
                lines = text.strip().splitlines()
                if lines and lines[0].startswith("```"):
                    # drop first fence line
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    # drop closing fence line
                    lines = lines[:-1]
                text = "\n".join(lines)
            # First attempt: direct JSON
            try:
                items = json.loads(text)
            except Exception:
                # Fallback: extract the outermost JSON array by bracket indices
                s = text.find("[")
                e = text.rfind("]")
                if s != -1 and e != -1 and e > s:
                    items = json.loads(text[s:e+1])
                else:
                    raise
            if not isinstance(items, list):
                return [], [], "Non-list JSON"
            titles: List[str] = []
            raw_items: List[dict] = []
            for it in items:
                # Accept either 'title' or 'name' to support different prompt schemas
                title = (it.get("title") or it.get("name") or "").strip()
                source = (it.get("source") or "").strip()
                url = (it.get("url") or "").strip()
                if title:
                    titles.append(title)
                    raw_items.append({"title": title, "source": source, "url": url, **{k: v for k, v in it.items() if k not in {"title", "source", "url"}}})
            return titles, raw_items, None
        except Exception:
            return [], [], "Invalid JSON in content"
    except Exception as e:
        return [], [], str(e)

def fetch_pplx_headlines_with_rotation(
    api_keys: List[str],
    prompt: str = DEFAULT_PROMPT,
    hours: int = 24,
    timeout: float = 12.0,
    backoff: float = 1.2
) -> Tuple[List[str], List[dict], Optional[str]]:
    keys = [k.strip() for k in api_keys if k and k.strip()]
    if not keys:
        return [], [], "No Perplexity API keys provided"

    last_err: Optional[str] = None
    # Allow overrides via env for robustness in different run environments
    try:
        timeout = float(os.getenv("TB_PPLX_TIMEOUT", str(timeout)))
    except Exception:
        pass
    try:
        backoff = float(os.getenv("TB_PPLX_BACKOFF", str(backoff)))
    except Exception:
        pass
    verbose = os.getenv("TB_UNDERRATED_VERBOSE", "0").lower() in ("1","true","on","yes")
    with httpx.Client(timeout=timeout) as client:
        for i, key in enumerate(keys):
            titles, items, err = _attempt_request(client, key, prompt, hours)
            # Treat as success if we got any items back, even if titles are empty
            if err is None and items:
                if verbose:
                    print(f"[pplx] key #{i+1}: success with {len(items)} items")
                return titles, items, None
            last_err = err or "Unknown error"
            if verbose:
                print(f"[pplx] key #{i+1}: warn: {last_err}")
            time.sleep(backoff * (i + 1))
    return [], [], last_err
