import os, requests
from dotenv import load_dotenv

load_dotenv(".env")
token = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
chat  = os.getenv("TELEGRAM_CHAT_ID","").strip()
assert token and chat, "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"

r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                  json={"chat_id": chat, "text": "Tracer Bullet: Telegram DM setup successful!", "disable_web_page_preview": True},
                  timeout=10)
print("status:", r.status_code, "ok:", r.ok, "body:", r.text[:200])
