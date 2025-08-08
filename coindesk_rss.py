import time
import requests
import xml.etree.ElementTree as ET
from typing import List

COINDESK_RSS = "https://www.coindesk.com/arc/outboundfeeds/rss/"

def _parse_titles(xml_text: str) -> List[str]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    titles = []
    for item in root.findall(".//item"):
        t = item.findtext("title") or ""
        t = t.strip()
        if t:
            titles.append(t)
    return titles

def fetch_coindesk_titles(timeout: float = 6.0, retries: int = 2, backoff: float = 1.5) -> List[str]:
    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.get(COINDESK_RSS, timeout=timeout, headers={"User-Agent": "TracerBot/1.0"})
            if r.status_code == 200:
                return _parse_titles(r.text)
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(backoff * (i + 1))
    return []
