import os
import requests

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Discord limits (approx)
MAX_EMBEDS = 10
MAX_TOTAL_CHARS = 6000
MAX_DESC_CHARS = 4096


def _embed_length(e: dict) -> int:
    desc_len = len(e.get("description", "") or "")
    title_len = len(e.get("title", "") or "")
    fields = e.get("fields", []) or []
    fields_len = 0
    for f in fields:
        fields_len += len((f.get("name") or "")) + len((f.get("value") or ""))
    return desc_len + title_len + fields_len


def _chunk_embeds(embeds):
    """Split embeds into lists of <=10 embeds and <=6000 chars total."""
    chunks = []
    current = []
    current_len = 0

    for e in embeds:
        e_len = _embed_length(e)
        if (len(current) >= MAX_EMBEDS) or (current_len + e_len > MAX_TOTAL_CHARS and current):
            chunks.append(current)
            current = []
            current_len = 0
        current.append(e)
        current_len += e_len
    if current:
        chunks.append(current)
    return chunks


def send_discord_digest_to(webhook_url: str, embeds):
    """Send embeds to a specific Discord webhook URL, respecting TB_ENABLE_DISCORD.

    Returns True on complete success, False otherwise.
    """
    # Evaluate enable flag at send time (not import time) so late-loaded .env is honored
    enable_discord = os.getenv("TB_ENABLE_DISCORD", "0") == "1"
    if not enable_discord or not webhook_url:
        print("[Discord] Disabled or missing webhook URL.")
        return False
    chunks = _chunk_embeds(embeds)
    ok_all = True
    for idx, chunk in enumerate(chunks, start=1):
        payload = {
            "content": f"[{idx}/{len(chunks)}]" if len(chunks) > 1 else None,
            "embeds": chunk,
        }
        try:
            resp = requests.post(webhook_url, json=payload, timeout=15)
        except Exception as e:
            print(f"[Discord] Exception: {e}")
            ok_all = False
            continue
        if resp.status_code not in (200, 201, 204):
            print(f"[Discord] Error {resp.status_code}: {resp.text}")
            ok_all = False
        else:
            print(f"[Discord] Sent part {idx}/{len(chunks)} with {len(chunk)} embeds.")
    return ok_all


def send_discord_digest(embeds):
    """Send digest via Discord webhook, splitting into multiple messages if needed."""
    return send_discord_digest_to(DISCORD_WEBHOOK_URL, embeds)
