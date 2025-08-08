import os
import httpx
from dotenv import load_dotenv

load_dotenv()

def collect_keys() -> list[str]:
    keys = []
    i = 1
    while True:
        v = os.getenv(f"PPLX_API_KEY_{i}", "")
        if not v:
            break
        keys.append(v.strip()); i += 1
    if not keys:
        raw = os.getenv("PPLX_API_KEYS", "")
        if raw:
            keys = [p.strip() for p in raw.split(",") if p.strip()]
        else:
            single = os.getenv("PPLX_API_KEY", "").strip()
            if single: keys = [single]
    return keys

URL = "https://api.perplexity.ai/chat/completions"
PAYLOAD = {
    "model": "sonar-pro",
    "messages": [{"role": "user", "content": "Return [] as JSON."}],
    "temperature": 0.0
}

keys = collect_keys()
print("testing_keys:", len(keys))
for idx, k in enumerate(keys, 1):
    try:
        r = httpx.post(URL, headers={"Authorization": f"Bearer {k}",
                                     "Content-Type": "application/json"},
                       json=PAYLOAD, timeout=10.0)
        print(f"key_{idx}_status:", r.status_code)
        if r.status_code != 200:
            print("body:", r.text[:200])
    except Exception as e:
        print(f"key_{idx}_error:", e)
