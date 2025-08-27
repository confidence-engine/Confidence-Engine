import os
import json
import time
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import httpx
from autocommit import auto_commit_and_push

from config import settings
from pplx_fetcher import fetch_pplx_headlines_with_rotation
from scripts.discord_sender import send_discord_digest_to
from telegram_bot import send_message as tg_send

# -----------------------------
# Config and constants
# -----------------------------

UND_STORE_PATH = os.getenv("TB_UNDERRATED_STORE", "data/underrated_store.json")
UND_OUT_DIR = os.getenv("TB_UNDERRATED_OUTDIR", "underrated_runs")

RUN_INTERVAL_DAYS = int(os.getenv("TB_UNDERRATED_RUN_INTERVAL_DAYS", "7"))
MCAP_THRESH_USD = float(os.getenv("TB_UNDERRATED_MARKETCAP_THRESHOLD", str(10_000_000)))
ALERT_DISCORD = os.getenv("TB_UNDERRATED_ALERT_DISCORD", "1") in ("1", "true", "on", "yes")
ALERT_TELEGRAM = os.getenv("TB_UNDERRATED_ALERT_TELEGRAM", "1") in ("1", "true", "on", "yes")
DISCORD_WEBHOOK = os.getenv("DISCORD_UNDERRATED_WEBHOOK_URL") or os.getenv("DISCORD_CRYPTO_SIGNALS_WEBHOOK_URL") or os.getenv("DISCORD_WEBHOOK_URL")

COINGECKO_API = os.getenv("TB_COINGECKO_API", "https://api.coingecko.com/api/v3")

PPLX_PROMPTS = [
    "List underrated crypto projects in 2025 with strong use cases but low market attention. Return JSON array of {name,ticker,desc,links}.",
    "What are promising blockchain projects with innovative tech yet to gain mainstream adoption? Return JSON array of {name,ticker,desc,links}.",
]

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Project:
    name: str
    ticker: str = ""
    desc: str = ""
    links: List[str] = None
    market_cap_usd: Optional[float] = None
    liquidity_score: Optional[float] = None
    score: Optional[float] = None
    timeline_months: Optional[str] = None
    why_matters: Optional[str] = None
    potential_impact: Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        # ensure lists
        d["links"] = self.links or []
        return d

# -----------------------------
# Utils: IO and persistence
# -----------------------------

def _ensure_dirs():
    os.makedirs(os.path.dirname(UND_STORE_PATH) or ".", exist_ok=True)
    os.makedirs(UND_OUT_DIR, exist_ok=True)

def _load_store() -> Dict[str, Any]:
    try:
        with open(UND_STORE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"alerted": {}, "last_run": None}

def _save_store(store: Dict[str, Any]):
    _ensure_dirs()
    with open(UND_STORE_PATH, "w") as f:
        json.dump(store, f, indent=2)

# -----------------------------
# Step 1: Discovery via Perplexity
# -----------------------------

def fetch_candidates() -> List[Project]:
    projects: Dict[str, Project] = {}
    keys = settings.pplx_api_keys or []
    for prompt in PPLX_PROMPTS:
        titles, items, err = fetch_pplx_headlines_with_rotation(keys, prompt=prompt, hours=168)
        if err:
            print(f"[underrated] PPLX warn: {err}")
        for it in items:
            name = (it.get("name") or it.get("title") or "").strip()
            if not name:
                # try to parse from title fallback
                continue
            ticker = (it.get("ticker") or "").strip().upper()
            desc = (it.get("desc") or it.get("description") or "").strip()
            links_raw = it.get("links") or []
            links = [str(x).strip() for x in (links_raw if isinstance(links_raw, list) else [links_raw]) if str(x).strip()]
            key = f"{name}|{ticker}"
            if key not in projects:
                projects[key] = Project(name=name, ticker=ticker, desc=desc, links=links)
            else:
                # merge simple fields
                p = projects[key]
                if desc and not p.desc:
                    p.desc = desc
                if links:
                    p.links = list({*(p.links or []), *links})
    return list(projects.values())

# -----------------------------
# Step 2: Enrichment
# -----------------------------

def _slugify_symbol(sym: str) -> str:
    return (sym or "").lower().replace(" ", "-")

def _coingecko_lookup(client: httpx.Client, name: str, ticker: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        # Search by name, fallback ticker by searching coins/list
        r = client.get(f"{COINGECKO_API}/search", params={"query": name}, timeout=15)
        if r.status_code == 200:
            j = r.json() or {}
            coins = j.get("coins") or []
            if coins:
                coin_id = coins[0].get("id")
                if coin_id:
                    r2 = client.get(f"{COINGECKO_API}/coins/markets", params={"vs_currency":"usd","ids":coin_id}, timeout=15)
                    if r2.status_code == 200 and isinstance(r2.json(), list) and r2.json():
                        d = r2.json()[0]
                        return float(d.get("market_cap") or 0.0), float(d.get("liquidity_score") or 0.0)
        # Fallback lightweight list by ticker match
        r3 = client.get(f"{COINGECKO_API}/coins/list", timeout=20)
        if r3.status_code == 200:
            coins = r3.json() or []
            for c in coins:
                if ticker and str(c.get("symbol","")) == ticker.lower():
                    r4 = client.get(f"{COINGECKO_API}/coins/markets", params={"vs_currency":"usd","ids":c.get("id")}, timeout=15)
                    if r4.status_code == 200 and r4.json():
                        d = r4.json()[0]
                        return float(d.get("market_cap") or 0.0), float(d.get("liquidity_score") or 0.0)
    except Exception as e:
        print(f"[coingecko] error: {e}")
    return None, None

# Removed GitHub and Twitter enrichment per request; only CoinGecko metrics are used.


def enrich_projects(candidates: List[Project]) -> List[Project]:
    out: List[Project] = []
    with httpx.Client(timeout=20) as client:
        for p in candidates:
            mc, liq = _coingecko_lookup(client, p.name, p.ticker)
            p.market_cap_usd = mc
            p.liquidity_score = liq
            out.append(p)
    return out

# -----------------------------
# Step 3: Filter and rank
# -----------------------------

def filter_and_rank(projects: List[Project]) -> List[Project]:
    ranked: List[Tuple[float, Project]] = []
    for p in projects:
        # basic thresholds
        if p.market_cap_usd is not None and p.market_cap_usd > MCAP_THRESH_USD:
            continue
        # Composite score: fundamentals (desc), liquidity, and small-cap preference
        fundamentals = 1.0 if (p.desc and len(p.desc) >= 40) else 0.6 if p.desc else 0.3
        liq = p.liquidity_score if isinstance(p.liquidity_score, (int, float)) else 0.0
        # Normalize liquidity roughly into [0,1]
        liq_norm = max(0.0, min(float(liq) / 1.0, 1.0)) if liq is not None else 0.3
        # Prefer smaller caps within threshold
        mcap = p.market_cap_usd or 0
        if mcap <= 5_000_000:
            mcap_factor = 1.0
        elif mcap <= 10_000_000:
            mcap_factor = 0.8
        elif mcap <= 50_000_000:
            mcap_factor = 0.6
        else:
            mcap_factor = 0.3
        score = 0.60 * fundamentals + 0.25 * liq_norm + 0.15 * mcap_factor
        p.score = round(score, 3)
        ranked.append((p.score, p))
    ranked.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in ranked]

# -----------------------------
# Step 4: Narratives
# -----------------------------

def generate_narratives(project: Project) -> Dict[str, str]:
    why = f"{project.name} aims to solve a meaningful problem with a clear use case: {project.desc[:180]}" if project.desc else f"{project.name} shows promising fundamentals and market potential."
    impact = "If attention grows, expect improved liquidity, listings, and potential ecosystem integrations as fundamentals and narrative align."
    # Timeline heuristic based on liquidity score (proxy for market readiness)
    liq = project.liquidity_score or 0
    if liq >= 0.8:
        tl = "3–6 months"
    elif liq >= 0.5:
        tl = "6–9 months"
    else:
        tl = "9–12 months"
    return {"why_matters": why, "potential_impact": impact, "timeline_months": tl}

# -----------------------------
# Step 5: Alert formatting
# -----------------------------

def _discord_embed_for(p: Project) -> dict:
    fields = []
    fields.append({"name": "Market cap", "value": f"${(p.market_cap_usd or 0):,.0f}" if p.market_cap_usd else "-", "inline": True})
    fields.append({"name": "Liquidity score", "value": f"{p.liquidity_score:.2f}" if isinstance(p.liquidity_score, (int, float)) else "-", "inline": True})
    fields.append({"name": "Estimated timeline", "value": p.timeline_months or "-", "inline": False})
    return {
        "title": f"{p.name} {('('+p.ticker+')') if p.ticker else ''}",
        "description": f"{(p.why_matters or '').strip()}\n\nPotential impact: {(p.potential_impact or '').strip()}",
        "fields": fields,
        "footer": {"text": f"Discovered by Agent via Perplexity API • {dt.datetime.utcnow().isoformat()}Z"},
        "color": 0x66CCFF,
    }


def _telegram_text_for(p: Project) -> str:
    parts = [
        f"<b>{p.name}</b> {('('+p.ticker+')') if p.ticker else ''}",
        (p.why_matters or "").strip(),
        "",
        f"Potential impact: {(p.potential_impact or '').strip()}",
        "",
        f"Market cap: {'$'+format(round(p.market_cap_usd)) if p.market_cap_usd else '-'} | Liquidity: {p.liquidity_score if p.liquidity_score is not None else '-'}",
        f"Estimated timeline: {p.timeline_months or '-'}",
        "",
        ("Links: " + ", ".join(p.links or [])) if p.links else "",
        "Discovered by Agent via Perplexity API",
    ]
    msg = "\n".join([x for x in parts if x is not None]).strip()
    return msg[:3800]

# -----------------------------
# Step 6: Dedup & send
# -----------------------------

def _already_alerted(store: Dict[str, Any], p: Project) -> bool:
    key = f"{p.name}|{p.ticker}".lower()
    return key in (store.get("alerted") or {})

def _mark_alerted(store: Dict[str, Any], p: Project):
    k = f"{p.name}|{p.ticker}".lower()
    store.setdefault("alerted", {})[k] = dt.datetime.utcnow().isoformat() + "Z"


def format_and_send_alerts(new_projects: List[Project]) -> Tuple[int, int]:
    sent_d, sent_t = 0, 0
    embeds = [_discord_embed_for(p) for p in new_projects]
    if ALERT_DISCORD and embeds and DISCORD_WEBHOOK:
        ok = send_discord_digest_to(DISCORD_WEBHOOK, embeds)
        sent_d = int(bool(ok))
    if ALERT_TELEGRAM and new_projects:
        for p in new_projects:
            ok = tg_send(_telegram_text_for(p))
            if ok:
                sent_t += 1
    return sent_d, sent_t

# -----------------------------
# Step 7: Outputs
# -----------------------------

def save_report(all_ranked: List[Project], alerted: List[Project]):
    _ensure_dirs()
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(UND_OUT_DIR, f"underrated_{ts}.json")
    out_md = os.path.join(UND_OUT_DIR, f"underrated_{ts}.md")
    data = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "projects": [p.to_dict() for p in all_ranked],
        "alerted": [p.to_dict() for p in alerted],
    }
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    lines = ["# Underrated crypto projects (auto)", ""]
    for i, p in enumerate(alerted, 1):
        lines.append(f"## {i}. {p.name} {('('+p.ticker+')') if p.ticker else ''}")
        lines.append((p.why_matters or "").strip())
        lines.append("")
        lines.append(f"Potential impact: {(p.potential_impact or '').strip()}")
        lines.append("")
        lines.append(f"Market cap: {'$'+format(round(p.market_cap_usd)) if p.market_cap_usd else '-'} | Liquidity: {p.liquidity_score if p.liquidity_score is not None else '-'}")
        lines.append(f"Timeline: {p.timeline_months or '-'}")
        if p.links:
            lines.append("Links: " + ", ".join(p.links))
        lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    # Optionally auto-commit artifacts
    do_ac = os.getenv("TB_UNDERRATED_GIT_AUTOCOMMIT", "1") in ("1", "true", "on", "yes")
    do_push = os.getenv("TB_UNDERRATED_GIT_PUSH", "1") in ("1", "true", "on", "yes")
    if do_ac:
        paths = [UND_OUT_DIR + "/", "data/underrated_store.json", "Dev_logs.md"]
        try:
            auto_commit_and_push(paths, extra_message="underrated scan artifacts", push_enabled=do_push)
        except Exception:
            pass

# -----------------------------
# Main
# -----------------------------

def main():
    store = _load_store()
    # interval guard
    try:
        last = store.get("last_run")
        if last:
            # Parse last as timezone-aware UTC
            last_dt = dt.datetime.fromisoformat(str(last).replace("Z", "+00:00"))
            now_utc = dt.datetime.now(dt.timezone.utc)
            # Ensure last_dt is aware; if naive, assume UTC
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=dt.timezone.utc)
            if now_utc - last_dt < dt.timedelta(days=RUN_INTERVAL_DAYS):
                print(f"[underrated] Skipping run due to interval guard ({RUN_INTERVAL_DAYS}d)")
                return
    except Exception:
        pass

    candidates = fetch_candidates()
    enriched = enrich_projects(candidates)
    ranked = filter_and_rank(enriched)

    # De-dupe by store
    fresh: List[Project] = []
    for p in ranked[:10]:  # send top N
        if not _already_alerted(store, p):
            fresh.append(p)
            p_updates = generate_narratives(p)
            p.why_matters = p_updates["why_matters"]
            p.potential_impact = p_updates["potential_impact"]
            p.timeline_months = p_updates["timeline_months"]

    if not fresh:
        print("[underrated] No new projects to alert.")
    else:
        sent_d, sent_t = format_and_send_alerts(fresh)
        print(f"[underrated] Alerts sent — Discord: {sent_d}, Telegram: {sent_t}")
        for p in fresh:
            _mark_alerted(store, p)

    store["last_run"] = dt.datetime.utcnow().isoformat() + "Z"
    _save_store(store)

    save_report(ranked, fresh)


if __name__ == "__main__":
    main()
