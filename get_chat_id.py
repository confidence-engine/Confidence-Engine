import os
import requests
from dotenv import load_dotenv

# Load .env from current directory explicitly
load_dotenv(".env")

token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not token:
    raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env first")

url = f"https://api.telegram.org/bot{token}/getUpdates"
r = requests.get(url, timeout=10)
print("status:", r.status_code)
print(r.text)  # Look for: "chat":{"id": <NUMBER>, "type":"private", ...}
