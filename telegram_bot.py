import os, json, traceback, requests
from dotenv import load_dotenv

load_dotenv(".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "Markdown")
API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def _escape_md(text: str) -> str:
    return (text.replace("_", "\\_").replace("*", "\\*").replace("[", "\\[").replace("`", "\\`"))

def send_message(text: str, chat_id: str = None, disable_web_page_preview: bool = True) -> bool:
    if not TELEGRAM_BOT_TOKEN:
        print("[Telegram] missing TELEGRAM_BOT_TOKEN"); return False
    cid = chat_id or TELEGRAM_CHAT_ID
    if not cid:
        print("[Telegram] missing TELEGRAM_CHAT_ID"); return False
    if TELEGRAM_PARSE_MODE.lower().startswith("markdown"):
        text = _escape_md(text)
    payload = {
        "chat_id": cid,
        "text": text,
        "disable_web_page_preview": disable_web_page_preview,
        "parse_mode": TELEGRAM_PARSE_MODE,
    }
    try:
        r = requests.post(f"{API_URL}/sendMessage", json=payload, timeout=10)
        if r.status_code != 200:
            print("[Telegram] sendMessage failed:", r.status_code, r.text[:200])
            return False
        return True
    except Exception:
        print("[Telegram] exception:\n", traceback.format_exc())
        return False

def format_alpha_message(payload: dict) -> str:
    symbol = payload.get("symbol", "BTC/USD")
    conf = float(payload.get("confidence", 0.0))
    gap = float(payload.get("divergence", 0.0))
    trig = float(payload.get("divergence_threshold", 1.0))
    vol_z = float(payload.get("volume_z", 0.0))
    action = payload.get("action", "HOLD")

    alpha_summary = payload.get("alpha_summary", "")
    alpha_next = payload.get("alpha_next_steps", "")

    try:
        used_heads = json.loads(payload.get("btc_filtered_headlines") or "[]")
    except Exception:
        used_heads = []

    try:
        rel = json.loads(payload.get("relevance_details") or "{}")
        acc = rel.get("accepted", [])
    except Exception:
        acc = []

    acc_sorted = sorted(acc, key=lambda x: x.get("weighted_relevance", 0.0), reverse=True)
    top = acc_sorted[:3]
    evid_lines = []
    for item in top:
        src = item.get("source", "?")
        wr = float(item.get("weighted_relevance", 0.0))
        h = (item.get("headline", "") or "")[:160]
        evid_lines.append(f"- [{src}] {wr:.3f} | {h}")

    header = f"Tracer Bullet • {symbol}"
    line1 = f"Action: {action} | Gap: {gap:+.2f} (trigger {trig:.2f}) | Conf: {conf:.2f} | VolZ: {vol_z:+.2f}"
    line2 = alpha_summary if alpha_summary else ""
    next_steps = alpha_next if alpha_next else ""
    evidence = "Evidence:\n" + "\n".join(evid_lines) if top else ""

    if TELEGRAM_PARSE_MODE.lower().startswith("markdown"):
        header = _escape_md(header)
        line1 = _escape_md(line1)
        line2 = _escape_md(line2)
        next_steps = _escape_md(next_steps)
        evidence = _escape_md(evidence)

    msg = f"{header}\n{line1}\n\n{line2}\n\n{next_steps}\n\n{evidence}".strip()
    return msg[:4000]
