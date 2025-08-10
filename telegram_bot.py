import os, json, traceback, requests, logging
from dotenv import load_dotenv

load_dotenv(".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "")
API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

logger = logging.getLogger(__name__)


def _escape_md(text: str) -> str:
    return (text.replace("_", "\\_").replace("*", "\\*").replace("[", "\\[").replace("`", "\\`"))

def send_message(text: str, chat_id: str = None, disable_web_page_preview: bool = True) -> bool:
    """
    Send a Telegram message using env-provided credentials.

    Honors TB_NO_TELEGRAM=1|true to skip sending (returns False, prints a note).
    No secrets are logged.
    """
    if os.getenv("TB_NO_TELEGRAM", "0").lower() in ("1", "true", "yes", "on"):
        logger.info("[Telegram] skipped due to TB_NO_TELEGRAM=1")
        return False
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("[Telegram] missing TELEGRAM_BOT_TOKEN"); return False
    cid = chat_id or TELEGRAM_CHAT_ID
    if not cid:
        logger.warning("[Telegram] missing TELEGRAM_CHAT_ID"); return False
    if TELEGRAM_PARSE_MODE.lower().startswith("markdown"):
        text = _escape_md(text)
    # Always truncate to stay under Telegram limit
    text = (text or "")[:4000]
    payload = {
        "chat_id": cid,
        "text": text,
        "disable_web_page_preview": disable_web_page_preview,
    }
    if TELEGRAM_PARSE_MODE:
        payload["parse_mode"] = TELEGRAM_PARSE_MODE
    try:
        r = requests.post(f"{API_URL}/sendMessage", json=payload, timeout=10)
        if r.status_code != 200:
            detail = ""
            try:
                j = r.json()
                detail = j.get("description") or r.text[:200]
            except Exception:
                detail = r.text[:200]
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After") or r.headers.get("retry-after")
                logger.warning(f"[Telegram] 429 Too Many Requests. Retry-After={retry_after}; detail={detail}")
            else:
                logger.warning(f"[Telegram] sendMessage failed: {r.status_code}; detail={detail}")
            return False
        return True
    except Exception:
        logger.error("[Telegram] exception:\n%s", traceback.format_exc())
        return False

def format_alpha_message(payload: dict) -> str:
    """
    Build a concise, alpha-first message from a payload without leaking secrets.
    Keeps within Telegram's 4096 char limit (we trim at 4000 to be safe).
    """
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

    # Diversity line if meaningful
    div = payload.get("source_diversity") or {}
    try:
        adj = float(div.get("adjustment", 0.0))
        uniq = int(div.get("unique", 0))
        top_share = float(div.get("top_source_share", 0.0))
    except Exception:
        adj, uniq, top_share = 0.0, 0, 0.0
    diversity_line = ""
    if abs(adj) >= 0.01:
        sign = "+" if adj >= 0 else ""
        diversity_line = f"Diversity adj: {sign}{adj:.2f} ({uniq} sources, top share {top_share:.2f})"

    if TELEGRAM_PARSE_MODE.lower().startswith("markdown"):
        header = _escape_md(header)
        line1 = _escape_md(line1)
        line2 = _escape_md(line2)
        next_steps = _escape_md(next_steps)
        evidence = _escape_md(evidence)
        diversity_line = _escape_md(diversity_line)

    parts = [header, line1, "", line2, "", next_steps]
    # Cascade line
    cas = payload.get("cascade_detector") or {}
    if (cas.get("tag") or "") == "HYPE_ONLY":
        rr = float(cas.get("repetition_ratio", 0.0) or 0.0)
        pm = float(cas.get("price_move_pct", 0.0) or 0.0)
        vzm = float(cas.get("max_volume_z", 0.0) or 0.0)
        cascade_line = f"Cascade: HYPE_ONLY (repetition {rr:.2f}, |Δp| {pm:.1f}%, VolZmax {vzm:.1f})"
        parts.extend(["", cascade_line])
    if diversity_line:
        parts.extend(["", diversity_line])
    if evidence:
        parts.extend(["", evidence])
    # Contrarian viewport
    cv = (payload.get("contrarian_viewport") or "").strip()
    if cv:
        parts.extend(["", f"Contrarian viewport: {cv}"])
    msg = "\n".join(parts).strip()
    return msg[:4000]
