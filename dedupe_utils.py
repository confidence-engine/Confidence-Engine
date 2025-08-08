import re
from typing import List

_whitespace_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^\w\s]")

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = _punct_re.sub(" ", s)
    s = _whitespace_re.sub(" ", s)
    return s.strip()

def dedupe_titles(titles: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in titles:
        if not isinstance(t, str):
            continue
        key = normalize_text(t)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(t.strip())
    return out
