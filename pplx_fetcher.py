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
    # Enforce recent results via web_search_options, targeting 'day' recency.
    # Include a time hint in the prompt too.
    return {
        "model": "sonar-pro",
        "messages": [
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
            items = json.loads(content)
            if not isinstance(items, list):
                return [], [], "Non-list JSON"
            titles: List[str] = []
            raw_items: List[dict] = []
            for it in items:
                title = (it.get("title") or "").strip()
                source = (it.get("source") or "").strip()
                url = (it.get("url") or "").strip()
                if title:
                    titles.append(title)
                    raw_items.append({"title": title, "source": source, "url": url})
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
    with httpx.Client(timeout=timeout) as client:
        for i, key in enumerate(keys):
            titles, items, err = _attempt_request(client, key, prompt, hours)
            if err is None and titles:
                return titles, items, None
            last_err = err or "Unknown error"
            time.sleep(backoff * (i + 1))
    return [], [], last_err
