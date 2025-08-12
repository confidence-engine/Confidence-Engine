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


def _split_text(text: str, max_len: int = 4000) -> list[str]:
    """Split text into chunks <= max_len, preferring logical boundaries.

    Preferred separators are used iteratively; if a resulting piece still
    exceeds max_len, we hard-split it. Trailing/leading whitespace trimmed.
    """
    # Preferred separators in descending order to keep sections whole
    separators = [
        "\n\nPlaybook",
        "\n\nMSFT", "\n\nAAPL", "\n\nSPY",
        "\n\nETH/", "\n\nBTC/",
        "\n\nEngine in One Minute",
        "\n\nWeekly Overview",
        "\n\nExecutive Take",
        "\n\n",  # generic paragraph split
    ]
    parts = [text]

    for sep in separators:
        new_parts = []
        for p in parts:
            if len(p) <= max_len:
                new_parts.append(p)
                continue
            # Split by separator and rebuild chunks under max_len
            chunks = p.split(sep)
            cur = ""
            for i, c in enumerate(chunks):
                piece = (sep + c) if i > 0 else c
                if len(cur) + len(piece) <= max_len:
                    cur += piece
                else:
                    if cur:
                        new_parts.append(cur)
                    if len(piece) > max_len:
                        # Hard-split very large piece
                        for j in range(0, len(piece), max_len):
                            new_parts.append(piece[j:j + max_len])
                        cur = ""
                    else:
                        cur = piece
            if cur:
                new_parts.append(cur)
        parts = new_parts

    # Final hard split safety
    final_parts = []
    for p in parts:
        if len(p) <= max_len:
            final_parts.append(p)
        else:
            for i in range(0, len(p), max_len):
                final_parts.append(p[i:i + max_len])

    return [s.strip() for s in final_parts if s.strip()]


def send_telegram_text_multi(text: str) -> bool:
    """Send long text by splitting into multiple messages.

    - Reuses env gating and retry logic of send_telegram_text.
    - Adds a header like "[i/N]" when multiple chunks are sent.
    """
    try:
        chunks = _split_text(text, max_len=4000)
        total = len(chunks)
        ok_all = True
        for idx, chunk in enumerate(chunks, start=1):
            header = f"[{idx}/{total}]\n" if total > 1 else ""
            ok = send_telegram_text(header + chunk)
            if not ok:
                ok_all = False
        return ok_all
    except Exception as e:
        print(f"[ERROR] send_telegram_text_multi failed: {e}")
        return False
