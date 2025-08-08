from typing import List, Tuple, Dict

def tag_origins(
    accepted: List[Tuple[str, float]],
    alpaca_heads: List[str],
    pplx_titles: List[str],
    coindesk_heads: List[str]
) -> List[Dict]:
    out = []
    alp = set(alpaca_heads)
    pplx = set(pplx_titles)
    cdx = set(coindesk_heads)
    for h, s in accepted:
        src = "unknown"
        if h in pplx: src = "perplexity"
        elif h in alp: src = "alpaca"
        elif h in cdx: src = "coindesk"
        out.append({"headline": h, "relevance": round(float(s), 3), "source": src})
    return out
