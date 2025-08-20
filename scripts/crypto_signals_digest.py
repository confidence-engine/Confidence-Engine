#!/usr/bin/env python3
"""
Standalone Crypto Signals Digest CLI (Discord-ready)
- Loads latest (or specified) universe JSON and builds a crypto-only digest.
- Renders Discord embeds via scripts.discord_formatter and can optionally send
  to a dedicated Discord channel webhook.

Safety defaults:
- Dry-run by default (prints summary to stdout, does NOT send).
- To send, pass --send and provide --webhook or DISCORD_CRYPTO_SIGNALS_WEBHOOK_URL.

Usage examples:
- Dry run (latest universe):
  python scripts/crypto_signals_digest.py

- Specify universe artifact and print markdown preview:
  python scripts/crypto_signals_digest.py --universe universe_runs/universe_20250811_080038_v31.json --print-md

- Send to a specific Discord webhook (crypto-signals channel):
  TB_ENABLE_DISCORD=1 \
  python scripts/crypto_signals_digest.py --send --webhook "https://discord.com/api/webhooks/..."

Env:
- DISCORD_CRYPTO_SIGNALS_WEBHOOK_URL (optional, used if --webhook not provided)
- TB_ENABLE_DISCORD=1 required to actually send when --send is used
- TB_DIGEST_MAX_COINS (default 6) limit of coins shown in Quick Summary
- TB_DIGEST_TOP_ALTS (optional) to control alt selection in future expansion
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root on path for intra-repo imports
_THIS_DIR = Path(__file__).parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# Lazy imports from scripts package
from scripts.discord_formatter import digest_to_discord_embeds  # type: ignore
from scripts.discord_sender import MAX_DESC_CHARS  # type: ignore
import requests
from alpaca import recent_bars as alpaca_recent_bars

try:
    # Reuse chunking post logic but allow custom webhook
    from scripts.discord_sender import _chunk_embeds  # type: ignore
except Exception:
    def _chunk_embeds(embeds):
        return [embeds]


def _load_dotenv_if_present():
    env_path = _PROJ_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text().splitlines():
            s = line.strip()
            if (not s) or s.startswith("#") or ("=" not in s):
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and (k not in os.environ or os.environ.get(k, "") == ""):
                os.environ[k] = v
    except Exception:
        pass


def _latest_universe_file() -> Optional[Path]:
    d = _PROJ_ROOT / "universe_runs"
    if not d.exists():
        return None
    cands = sorted([p for p in d.glob("universe_*.json") if p.is_file()])
    return cands[-1] if cands else None


def _is_crypto_symbol(sym: str) -> bool:
    if not sym:
        return False
    s = str(sym).upper()
    return ("/" in s and s.endswith(("/USD", "/USDT", "/USDC"))) or any(
        s.startswith(pfx) for pfx in ("BTC/", "ETH/", "SOL/", "XRP/", "ADA/", "BNB/", "DOGE/", "AVAX/", "LTC/", "DOT/")
    )


def _coerce_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _infer_thesis(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal, number-free, consistent with Discord formatter expectations
    div = payload.get("divergence") or 0.0
    conf = payload.get("confidence") or 0.0
    try:
        div = float(div)
        conf = float(conf)
    except Exception:
        div = 0.0
        conf = 0.0
    bias = "up" if div > 0 else ("down" if div < 0 else "neutral")
    readiness = "Now" if conf > 0.75 else ("Near" if conf > 0.65 else "Later")
    # Risk proxy from position_sizing.target_R if present
    ps = payload.get("position_sizing") or {}
    try:
        r = float(ps.get("target_R", 0.0) or 0.0)
    except Exception:
        r = 0.0
    risk_band = "High" if r > 0.6 else ("Medium" if r > 0.3 else "Low")
    action = (
        "Buy" if (div > 0 and conf > 0.7) else ("Sell" if (div < 0 and conf > 0.7) else "Watch")
    )
    return {"bias": bias, "readiness": readiness, "risk_band": risk_band, "action": action}


def _asset_from_payload(p: Dict[str, Any]) -> Dict[str, Any]:
    th = _infer_thesis(p)
    out: Dict[str, Any] = {
        "symbol": p.get("symbol"),
        "structure": p.get("structure") or p.get("pattern"),
        "thesis": th,
        # pass-through popular flags used by formatter
        "alignment_flag": p.get("alignment_flag"),
        "timescale_scores": p.get("timescale_scores") or {},
        "confirmation_checks": p.get("confirmation_checks") or [],
        "participation": p.get("participation") or p.get("flow"),
        # optional: plan snapshot if artifact was enriched previously
        "plan": p.get("plan") or {},
    }
    # If artifact uses legacy keys (levels with entry/invalid/targets) and no plan snapshot,
    # leave plan empty; formatter will still render evidence lines and header.
    return out


def _build_digest_data(universe: Dict[str, Any], max_coins: int) -> Dict[str, Any]:
    payloads = universe.get("payloads") or []
    crypto_payloads = [p for p in payloads if _is_crypto_symbol(str(p.get("symbol", "")))]
    # Order: BTC, ETH, others in given order
    btc = [p for p in crypto_payloads if str(p.get("symbol", "")).upper().startswith("BTC/")]
    eth = [p for p in crypto_payloads if str(p.get("symbol", "")).upper().startswith("ETH/")]
    others = [p for p in crypto_payloads if p not in btc + eth]
    ordered = btc + eth + others

    max_n = int(os.getenv("TB_DIGEST_MAX_COINS", str(max_coins)))
    # Enrich ordered crypto payloads with per-TF plan like tracer_bullet_universe
    ORDERED_TFS = ["1h", "4h", "1D", "1W"]

    def _strength_from_scores(p: dict, tf: str) -> float:
        ts = p.get("timescale_scores") or {}
        key = {"1h": "short", "4h": "mid", "1D": "mid", "1W": "long", "1M": "long"}.get(tf)
        try:
            val = ts.get(key)
            if isinstance(val, dict):
                v = float(val.get("divergence", 0.0))
            else:
                v = float(val or 0.0)
        except Exception:
            v = 0.0
        m = max(0.0, min(1.0, abs(v)))
        return 0.5 + m

    def _compose_why(tf: str, direction: int, struct_hint: str, pattern_hint: str, strength: float, readiness: str, action: str, payload: dict, thesis: dict) -> str:
        tf_desc = {"1h": "intraday", "4h": "swing", "1D": "daily", "1W": "weekly", "1M": "monthly"}.get(tf, tf)
        if direction > 0:
            why = f"Bias up on {tf} ({tf_desc}); using trigger entries aligned with momentum ({struct_hint})."
        elif direction < 0:
            why = f"Bias down on {tf} ({tf_desc}); using fade entries into structure ({struct_hint})."
        else:
            why = f"Neutral bias on {tf} ({tf_desc}); tight trigger and symmetric guard ({struct_hint})."
        if pattern_hint:
            why += f" {pattern_hint}."
        strength_lbl = ("elevated" if strength > 1.15 else ("balanced" if strength > 0.85 else "muted"))
        why += f" Strength {strength_lbl}, timing {readiness}, action {action}."
        try:
            wk = payload.get("weekly_anchor") or {}
            if isinstance(wk, dict):
                if wk.get("supply_zone") and direction <= 0:
                    why += " Weekly: into supply."
                if wk.get("demand_zone") and direction >= 0:
                    why += " Weekly: from demand."
        except Exception:
            pass
        try:
            aligned = bool(payload.get("alignment_flag") or thesis.get("tf_aligned") or False)
            if tf in ("1h",):
                band = "short"
            elif tf in ("4h", "1D"):
                band = "mid"
            else:
                band = "long"
            z = None
            tss = payload.get("timescale_scores", {})
            if isinstance(tss.get(band), dict):
                z = tss.get(band, {}).get("volume_z")
            flow = None
            try:
                from explain import volume_label
                flow = volume_label(z) if isinstance(z, (int, float)) else None
            except Exception:
                flow = None
            conf_pen = payload.get("confirmation_penalty")
            conf_checks = payload.get("confirmation_checks")
            if conf_checks is not None:
                if isinstance(conf_pen, (int, float)) and conf_pen < 0:
                    conf_txt = "confirmation weak"
                else:
                    conf_txt = "confirmation OK"
            else:
                conf_txt = "confirmation pending"
            parts = []
            parts.append("aligned" if aligned else "mixed alignment")
            if flow:
                parts.append(flow)
            if conf_txt:
                parts.append(conf_txt)
            if parts:
                why += " Signals: " + ", ".join(parts) + "."
        except Exception:
            pass
        return why

    def _map_to_binance(sym: str) -> str:
        s = (sym or "").upper()
        base, _, quote = s.partition("/")
        q = quote or "USDT"
        if q in ("USD", "USDC"):
            q = "USDT"
        return f"{base}{q}"

    def binance_spot_price(sym: str):
        try:
            bsym = _map_to_binance(sym)
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={bsym}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return float(r.json().get("price"))
        except Exception:
            pass
        return None

    def _normalize_crypto_symbol(sym: str) -> str:
        if "/" in sym:
            return sym
        if len(sym) >= 6:
            base = sym[:-3]
            quote = sym[-3:]
            return f"{base}/{quote}"
        return sym

    def alpaca_spot_price(sym: str):
        try:
            norm = _normalize_crypto_symbol(sym)
            df = alpaca_recent_bars(norm, minutes=5)
            if df is not None and len(df) > 0 and "close" in df.columns:
                return float(df["close"].iloc[-1])
        except Exception:
            pass
        return None

    def get_crypto_spot_price(symbol: str):
        price = binance_spot_price(symbol)
        if price:
            return price
        price = alpaca_spot_price(symbol)
        if price:
            return price
        return None

    def synthesize_analysis_levels(sym: str, p: dict, spot: float, thesis: dict) -> dict:
        if not spot:
            return {}
        bias = str(thesis.get("bias", "")).lower()
        direction = 1 if bias == "up" else (-1 if bias == "down" else 0)
        readiness = str(thesis.get("readiness", "")).lower() or "later"
        action = str(thesis.get("action", "")).lower() or "watch"
        struct_txt = str((p.get("structure") or p.get("pattern") or "")).lower()
        if "breakout" in struct_txt or "bo" in struct_txt:
            struct_hint = "breakout context"
        elif "trend" in struct_txt:
            struct_hint = "trend continuation"
        elif "range" in struct_txt or "reversion" in struct_txt:
            struct_hint = "range context"
        else:
            struct_hint = "neutral structure"
        if any(k in struct_txt for k in ("breakout", "bo", "b/o")):
            pattern_hint = "breakout setup"
        elif "retest" in struct_txt:
            pattern_hint = "retest setup"
        elif "pullback" in struct_txt or "pb" in struct_txt:
            pattern_hint = "pullback setup"
        elif "range" in struct_txt:
            pattern_hint = "range fade"
        else:
            pattern_hint = None
        base = {
            "1h":   {"trig": 0.003, "fade_hi": 0.007, "fade_lo": 0.004, "inv": 0.008, "tp1": 0.007, "tp2": 0.015},
            "4h":   {"trig": 0.005, "fade_hi": 0.010, "fade_lo": 0.006, "inv": 0.015, "tp1": 0.012, "tp2": 0.025},
            "1D":   {"trig": 0.008, "fade_hi": 0.015, "fade_lo": 0.010, "inv": 0.020, "tp1": 0.020, "tp2": 0.040},
            "1W":   {"trig": 0.015, "fade_hi": 0.030, "fade_lo": 0.020, "inv": 0.035, "tp1": 0.040, "tp2": 0.080},
        }
        out = {}
        for tf in ORDERED_TFS:
            pcts = dict(base[tf])
            strength = _strength_from_scores(p, tf)
            for k in pcts:
                pcts[k] *= strength
            why = _compose_why(tf, direction, struct_hint, pattern_hint, strength, readiness, action, p, thesis)
            if direction > 0:
                out[tf] = {
                    "entry_type": "trigger",
                    "entries": spot * (1 + pcts["trig"]),
                    "invalidation": {"price": spot * (1 - pcts["inv"]), "condition": "close below"},
                    "targets": [
                        {"label": "TP1", "price": spot * (1 + pcts["tp1"])},
                        {"label": "TP2", "price": spot * (1 + pcts["tp2"])},
                    ],
                    "source": "analysis",
                    "explain": why,
                }
            elif direction < 0:
                out[tf] = {
                    "entry_type": "fade",
                    "entries": [spot * (1 + pcts["fade_lo"]), spot * (1 + pcts["fade_hi"])],
                    "invalidation": {"price": spot * (1 + pcts["inv"]), "condition": "close above"},
                    "targets": [
                        {"label": "TP1", "price": spot * (1 - pcts["tp1"])},
                        {"label": "TP2", "price": spot * (1 - pcts["tp2"])},
                    ],
                    "source": "analysis",
                    "explain": why,
                }
            else:
                out[tf] = {
                    "entry_type": "trigger",
                    "entries": spot * (1 + 0.001 * (1 if strength >= 1.0 else -1)),
                    "invalidation": {"price": spot * (1 - 0.004), "condition": "breach"},
                    "targets": [
                        {"label": "TP1", "price": spot * (1 + 0.006)},
                        {"label": "TP2", "price": spot * (1 + 0.012)},
                    ],
                    "source": "analysis",
                    "explain": why,
                }
        return out

    def build_tf_plan_from_levels(levels: dict):
        if not levels:
            return {}
        entries_out = []
        invalidation_out = None
        targets_out = []
        if isinstance(levels.get("entries"), (int, float)):
            entries_out.append({"type": levels.get("entry_type") or "trigger", "zone_or_trigger": float(levels["entries"])})
        elif isinstance(levels.get("entries"), (list, tuple)):
            ent = levels.get("entries")
            if len(ent) == 2 and all(isinstance(x, (int, float)) for x in ent):
                lo, hi = ent
                entries_out.append({"type": levels.get("entry_type") or "fade", "zone_or_trigger": [float(lo), float(hi)]})
        inv = levels.get("invalidation")
        if isinstance(inv, dict) and isinstance(inv.get("price"), (int, float)):
            invalidation_out = {"price": float(inv["price"]), "condition": inv.get("condition", "close below")}
        if isinstance(levels.get("targets"), (list, tuple)):
            for i, tp in enumerate(levels.get("targets") or [], start=1):
                if isinstance(tp, (int, float)):
                    targets_out.append({"label": f"TP{i}", "price": float(tp)})
                elif isinstance(tp, dict) and isinstance(tp.get("price"), (int, float)):
                    targets_out.append({"label": tp.get("label") or f"TP{i}", "price": float(tp["price"])})
        if not entries_out and not invalidation_out and not targets_out:
            return None
        explain = levels.get("explain") if isinstance(levels, dict) else None
        plan = {"entries": entries_out, "invalidation": invalidation_out, "targets": targets_out}
        if explain:
            plan["explain"] = str(explain)
        return plan

    def collect_tf_levels(symbol: str, tf: str, analysis_map: dict):
        per_tf = (analysis_map.get("levels") or {}).get(tf)
        return build_tf_plan_from_levels(per_tf)

    def build_plan_all_tfs(symbol: str, analysis_map: dict):
        plan = {}
        for tf in ORDERED_TFS:
            tf_plan = collect_tf_levels(symbol, tf, analysis_map)
            if tf_plan:
                tf_plan["source"] = "analysis"
                plan[tf] = tf_plan
        return plan

    # Attach plan snapshots
    for p in ordered:
        sym = p.get("symbol")
        if not sym:
            continue
        thesis = _infer_thesis(p)
        spot_price = get_crypto_spot_price(sym)
        try:
            if not isinstance(p.get("levels"), dict) and spot_price:
                p["levels"] = synthesize_analysis_levels(sym, p, spot_price, thesis)
        except Exception:
            pass
        plan = build_plan_all_tfs(sym, p)
        if plan:
            p["plan"] = plan

    assets = [_asset_from_payload(p) for p in ordered[:max_n]]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    weekly_text = "Crypto snapshot: trade only clean setups; respect risk; be patient in chop."
    engine_text = {
        "thesis_text": universe.get("engine_thesis") or "Engine: focus on alignment, confirmation, and clear structure.",
        "evidence_bullets": [
            "Avoid chasing; wait for clean triggers.",
            "Add only after confirmation; keep risk small in mixed tape.",
        ],
        "action_hint": "Prefer A-setups; pass on messy charts.",
    }
    return {
        "timestamp": ts,
        "executive_take": universe.get("executive_take") or "Clear plan. Keep it simple and safe.",
        "weekly": {"regime": universe.get("weekly_regime"), "plan_text": weekly_text},
        "engine": engine_text,
        "assets": assets,
        # No polymarket section here; this is crypto-only signals.
    }


def _print_preview(embeds: List[Dict[str, Any]], as_md: bool = False) -> None:
    if as_md:
        print("# Crypto Signals — Preview\n")
        for e in embeds:
            title = e.get("title") or "-"
            print(f"## {title}")
            desc = e.get("description") or ""
            if desc:
                print(desc)
            for f in (e.get("fields") or []):
                print(f"- **{f.get('name','-')}**: {f.get('value','-')}")
            print("")
    else:
        for i, e in enumerate(embeds, 1):
            print(f"[{i}] {e.get('title','-')}")
            desc = (e.get("description") or "")
            if desc:
                print(desc[:MAX_DESC_CHARS])
            for f in (e.get("fields") or []):
                print(f" - {f.get('name','-')}: {f.get('value','-')}")
            print("-")


def _post_to_discord(embeds: List[Dict[str, Any]], webhook_url: str) -> bool:
    import requests
    chunks = _chunk_embeds(embeds)
    ok_all = True
    for idx, chunk in enumerate(chunks, start=1):
        payload = {"content": f"[crypto-signals {idx}/{len(chunks)}]" if len(chunks) > 1 else None, "embeds": chunk}
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
            print(f"[Discord] Sent part {idx}/{len(chunks)} with {len(chunk)} embeds to crypto-signals.")
    return ok_all


def main() -> int:
    _load_dotenv_if_present()
    ap = argparse.ArgumentParser(description="Crypto Signals digest (Discord)")
    ap.add_argument("--universe", default=None, help="Universe JSON to read; default: latest in universe_runs/")
    ap.add_argument("--max-coins", type=int, default=6, help="Max number of coins to include (default: 6)")
    ap.add_argument("--print-md", action="store_true", help="Print a markdown preview to stdout")
    ap.add_argument("--send", action="store_true", help="Actually send to Discord webhook (default: dry-run)")
    ap.add_argument("--webhook", default=None, help="Discord webhook URL for crypto-signals channel")
    args = ap.parse_args()

    uni_path = Path(args.universe) if args.universe else _latest_universe_file()
    if not uni_path or not uni_path.exists():
        print("[crypto_signals_digest] No universe file found. Run the universe scanner first.")
        return 1
    try:
        with open(uni_path, "r", encoding="utf-8") as f:
            universe = json.load(f)
    except Exception as e:
        print(f"[crypto_signals_digest] Failed to read universe file: {e}")
        return 1

    digest = _build_digest_data(universe, max_coins=args.max_coins)
    # Add provenance so header shows Source: artifact @ gitsha
    try:
        import subprocess
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(_PROJ_ROOT)).decode().strip()
    except Exception:
        git_sha = None
    digest["provenance"] = {"artifact": uni_path.name, "git": git_sha}
    # Ensure Polymarket sections are suppressed for crypto digest
    os.environ.setdefault("TB_POLYMARKET_SECTION", "0")
    os.environ.setdefault("TB_POLYMARKET_SHOW_EMPTY", "0")
    embeds = digest_to_discord_embeds(digest)

    # If --send not provided, allow auto-send via env for seamless runs
    if not args.send:
        if os.getenv("TB_CRYPTO_DIGEST_AUTOSEND", "0") == "1":
            args.send = True

    # Preview
    if args.print_md:
        _print_preview(embeds, as_md=True)
    else:
        _print_preview(embeds, as_md=False)

    # Safe-by-default: only send if --send provided and TB_ENABLE_DISCORD=1
    webhook = args.webhook or os.getenv("DISCORD_CRYPTO_SIGNALS_WEBHOOK_URL") or ""
    if args.send:
        # Safety gate remains; can be set once in .env for seamless runs
        if os.getenv("TB_ENABLE_DISCORD", "0") != "1":
            print("[crypto_signals_digest] TB_ENABLE_DISCORD!=1; not sending (safety gate) — set in .env for seamless sends")
            return 0
        if not webhook:
            print("[crypto_signals_digest] --webhook or DISCORD_CRYPTO_SIGNALS_WEBHOOK_URL is required when --send is used")
            return 2
        ok = _post_to_discord(embeds, webhook)
        return 0 if ok else 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
