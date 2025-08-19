#!/usr/bin/env python3
"""
Standalone Polymarket Digest CLI
- Fetches Polymarket crypto markets and renders a simple report.
- Defaults to the native provider (`providers/polymarket.py`) which uses the public API.
- Optionally use the Perplexity-backed provider (`providers/polymarket_pplx.py`).

Env (common):
- TB_POLYMARKET_ASSETS (e.g., "BTC,ETH,SOL").
- TB_POLYMARKET_LIMIT (hints to provider; native appends as query param; PPLX may follow prompt limit).
- TB_POLYMARKET_DEBUG=1 to print provider debug.

PPLX provider env:
- PPLX_API_KEY[_N]
- PPLX_TIMEOUT

Usage examples:
- python scripts/polymarket_digest.py --full
- python scripts/polymarket_digest.py --provider pplx --format md --output polymarket.md
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List
from datetime import datetime, timezone

# Ensure project root is in sys.path for `providers.*` imports when running as a script
_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_this_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# Import providers lazily within functions to avoid side-effects at import time


def _normalize_native(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in items or []:
        title = m.get("title") or m.get("question") or m.get("name") or m.get("slug") or ""
        end_date = (
            m.get("endDate")
            or m.get("end_date")
            or m.get("closesAt")
            or m.get("endTime")
            or m.get("closeTime")
            or m.get("expiry")
        )
        # liquidity detection
        liq = None
        for k in ("liquidity", "liquidityUSD", "liquidity_usd", "volume24h", "volume", "totalVolumeUSD"):
            v = m.get(k)
            if isinstance(v, (int, float)):
                liq = float(v)
                break
        # implied price/prob (best-effort)
        prob = None
        for k in ("impliedProbability", "implied_probability", "yesPrice", "price", "p_yes", "probability"):
            v = m.get(k)
            try:
                if v is not None:
                    x = float(v)
                    if x > 1.0 and x <= 100.0:
                        x = x / 100.0
                    if 0.0 <= x <= 1.0:
                        prob = x
                        break
            except Exception:
                pass
        out.append({
            "title": str(title),
            "endDate": end_date,
            "liquidityUSD": liq,
            "impliedProbability": prob,
            "resolutionSource": m.get("resolutionSource") or m.get("resolution_source") or "",
            "marketId": m.get("conditionId") or m.get("id") or m.get("slug") or m.get("questionId"),
            "asset": _infer_asset(title),
        })
    return out


def _infer_asset(title: str) -> str:
    t = (title or "").lower()
    if "bitcoin" in t or "btc" in t:
        return "BTC"
    if "ethereum" in t or "eth" in t:
        return "ETH"
    if "solana" in t or "sol" in t:
        return "SOL"
    if "xrp" in t or "ripple" in t:
        return "XRP"
    return "OTHER"


def _fetch_native(full: bool, min_liq: float, min_weeks: int, max_weeks: int) -> List[Dict[str, Any]]:
    from providers.polymarket import get_btc_eth_markets

    max_items = None if full else int(os.getenv("TB_POLYMARKET_MAX_ITEMS", "6"))
    return get_btc_eth_markets(
        min_liquidity=min_liq,
        min_weeks=min_weeks,
        max_weeks=max_weeks,
        max_items=max_items,
    )


def _fetch_pplx(full: bool) -> List[Dict[str, Any]]:
    from providers.polymarket_pplx import get_crypto_markets_via_pplx

    # PPLX provider already returns a normalized-like schema with keys we render below.
    limit_env = os.getenv("TB_POLYMARKET_LIMIT")
    limit = int(limit_env) if (limit_env or "").isdigit() else None
    items = get_crypto_markets_via_pplx(limit=limit)
    # Cannot truly be "full" because pplx provider enforces a top-6; we just pass through
    return items


def _render_text(items: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("Polymarket — Standalone Digest")
    lines.append("".ljust(32, "-"))
    if not items:
        lines.append("No qualifying markets found.")
        return "\n".join(lines)
    for i, m in enumerate(items, 1):
        title = m.get("title") or "(untitled)"
        asset = m.get("asset") or ""
        end_date = m.get("endDate") or ""
        liq = m.get("liquidityUSD")
        prob = m.get("impliedProbability")
        rs = m.get("resolutionSource") or ""
        lines.append(f"{i}. {title}")
        meta = []
        if asset:
            meta.append(asset)
        if end_date:
            meta.append(f"ends {end_date}")
        if liq is not None:
            meta.append(f"liq ${liq:,.0f}")
        if prob is not None:
            meta.append(f"p_yes {prob*100:.1f}%")
        if rs:
            meta.append(f"res: {rs}")
        if meta:
            lines.append("   - " + "; ".join(meta))
    return "\n".join(lines)


def _render_md(items: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("# Polymarket — Standalone Digest\n")
    if not items:
        lines.append("_No qualifying markets found._")
        return "\n".join(lines)
    for m in items:
        title = m.get("title") or "(untitled)"
        asset = m.get("asset") or ""
        end_date = m.get("endDate") or ""
        liq = m.get("liquidityUSD")
        prob = m.get("impliedProbability")
        rs = m.get("resolutionSource") or ""
        lines.append(f"- **{title}**")
        details = []
        if asset:
            details.append(f"asset: `{asset}`")
        if end_date:
            details.append(f"ends: `{end_date}`")
        if liq is not None:
            details.append(f"liq: `${liq:,.0f}`")
        if prob is not None:
            details.append(f"p_yes: `{prob*100:.1f}%`")
        if rs:
            details.append(f"resolution: `{rs}`")
        if details:
            lines.append("  - " + "; ".join(details))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone Polymarket digest generator")
    ap.add_argument("--provider", choices=["native", "pplx"], default=os.getenv("TB_POLYMARKET_PROVIDER", "pplx"), help="Market provider to use (default: pplx)")
    ap.add_argument("--full", action="store_true", help="No local cap on number of items (native provider only; PPLX controlled via TB_POLYMARKET_PPLX_MAX)")
    ap.add_argument("--format", choices=["text", "md", "json"], default="text", help="Output format")
    ap.add_argument("--output", "-o", default=None, help="Output file path (writes to stdout if omitted)")
    ap.add_argument("--min-liq", type=float, default=float(os.getenv("TB_POLYMARKET_MIN_LIQ", "1000")), help="Minimum liquidity filter (native)")
    ap.add_argument("--min-weeks", type=int, default=int(os.getenv("TB_POLYMARKET_MIN_WEEKS", "1")), help="Earliest end window (native)")
    ap.add_argument("--max-weeks", type=int, default=int(os.getenv("TB_POLYMARKET_MAX_WEEKS", "12")), help="Latest end window (native)")
    args = ap.parse_args()

    if args.provider == "native":
        raw = _fetch_native(full=args.full, min_liq=args.min_liq, min_weeks=args.min_weeks, max_weeks=args.max_weeks)
        items = _normalize_native(raw)
    else:
        items = _fetch_pplx(full=args.full)

    if args.format == "json":
        out = json.dumps(items, indent=2)
    elif args.format == "md":
        out = _render_md(items)
    else:
        out = _render_text(items)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out + ("\n" if not out.endswith("\n") else ""))
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
