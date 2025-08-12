#!/usr/bin/env python3
"""
Lightweight Telegram sender gated by environment flags.
"""
import os
import time
from typing import Optional

import requests


def send_telegram_text(text: str) -> bool:
    """
    Send plain text to Telegram. Returns True if a 2xx response was received.
    - If TB_NO_TELEGRAM=1 or token/chat missing: return False without sending.
    - Retries up to 3 times with backoff; respects Retry-After on 429.
    """
    if os.getenv("TB_NO_TELEGRAM", "0") == "1":
        return False
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TB_TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TB_TELEGRAM_CHAT_ID")
    if not token or not chat_id or not text:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}

    delays = [0.5, 1.0, 2.0]
    for attempt, delay in enumerate(delays, start=1):
        try:
            r = requests.post(url, json=data, timeout=15)
            if 200 <= r.status_code < 300:
                return True
            if r.status_code == 429:
                retry_after = 0
                try:
                    body = r.json()
                    retry_after = int(body.get("parameters", {}).get("retry_after", 0))
                except Exception:
                    pass
                # Fallback to Retry-After header if present
                try:
                    if not retry_after:
                        ra_hdr = r.headers.get("Retry-After")
                        if ra_hdr:
                            retry_after = int(ra_hdr)
                except Exception:
                    pass
                time.sleep(max(retry_after, delay))
                continue
            # Other non-2xx: wait and retry
            time.sleep(delay)
        except Exception:
            time.sleep(delay)
    return False
