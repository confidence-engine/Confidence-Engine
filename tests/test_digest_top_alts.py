import yaml
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tracer_bullet_universe import select_digest_symbols_public


def test_config_contains_btc_eth_and_alts():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "universe.yaml")
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    crypto = cfg.get("crypto") or []
    assert "BTC/USD" in crypto
    assert "ETH/USD" in crypto
    # Ensure we have at least some alts appended
    alts = [s for s in crypto if s not in ("BTC/USD", "ETH/USD")]
    assert len(alts) >= 5


def test_digest_surfaces_btc_eth_plus_top_k_alts():
    # Build a ranked payloads list: BTC, ETH, then 6 alts
    def _p(sym):
        return {"symbol": sym, "symbol_type": "crypto"}

    payloads_ranked = [
        _p("BTC/USD"),
        _p("ETH/USD"),
        _p("SOL/USD"),
        _p("BNB/USD"),
        _p("XRP/USD"),
        _p("ADA/USD"),
        _p("DOGE/USD"),
        _p("LINK/USD"),
    ]

    # K = 3 → expect BTC, ETH, first 3 alts from ranking
    sel = select_digest_symbols_public(payloads_ranked, 3)
    assert sel == ["BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD", "XRP/USD"]

    # K = 0 → only BTC and ETH
    sel0 = select_digest_symbols_public(payloads_ranked, 0)
    assert sel0 == ["BTC/USD", "ETH/USD"]

    # K larger than available → include all alts
    sel_max = select_digest_symbols_public(payloads_ranked, 20)
    assert sel_max == [
        "BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD", "XRP/USD", "ADA/USD", "DOGE/USD", "LINK/USD"
    ]
