#!/usr/bin/env python3
"""
Universe scanner for multi-asset analysis.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv

from bars_stock import get_bars_stock
from digest_utils import format_universe_digest, validate_digest_length, truncate_digest
from scripts.digest_formatter import render_digest
from symbol_utils import normalize_symbol, get_symbol_type, validate_universe_config
from trading_hours import trading_hours_state
import tracer_bullet
import telegram_bot


def load_universe_config(config_path: str) -> Dict[str, List[str]]:
    """
    Load universe configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with "crypto" and "stocks" lists
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not validate_universe_config(config):
            raise ValueError("Invalid universe configuration")
        
        return config
    except Exception as e:
        print(f"Error loading universe config: {e}")
        sys.exit(1)


def get_bars_for_symbol(symbol: str, lookback_minutes: int) -> List[Dict]:
    """
    Get bars data for a symbol based on its type.
    
    Args:
        symbol: Symbol string
        lookback_minutes: Number of minutes to look back
        
    Returns:
        List of bar dictionaries
    """
    symbol_type = get_symbol_type(symbol)
    
    if symbol_type == "crypto":
        # Use existing crypto bars functionality
        # This would need to be refactored from tracer_bullet.py
        # For now, we'll use a placeholder
        return _get_crypto_bars_placeholder(symbol, lookback_minutes)
    elif symbol_type == "stock":
        return get_bars_stock(symbol, lookback_minutes)
    else:
        raise ValueError(f"Unknown symbol type for {symbol}")


def _get_crypto_bars_placeholder(symbol: str, lookback_minutes: int) -> List[Dict]:
    """
    Placeholder for crypto bars - this should be refactored from tracer_bullet.py.
    
    Args:
        symbol: Crypto symbol
        lookback_minutes: Number of minutes to look back
        
    Returns:
        List of bar dictionaries
    """
    # For now, return stub data
    # In production, this would call the actual crypto bars function
    import random
    from datetime import datetime, timezone, timedelta
    
    bars = []
    now = datetime.now(timezone.utc)
    
    base_price = 50000.0 if symbol == "BTC/USD" else 3000.0
    
    for i in range(lookback_minutes):
        timestamp = now - timedelta(minutes=i)
        ts = int(timestamp.timestamp())
        
        random.seed(hash(symbol) + i)
        price_change = (random.random() - 0.5) * 0.02
        close = base_price * (1 + price_change)
        volume = 1000000 * (0.5 + random.random())
        
        bars.append({
            "ts": ts,
            "close": round(close, 2),
            "volume": int(volume)
        })
    
    bars.reverse()
    return bars


def analyze_symbol(symbol: str, lookback_minutes: int, env: Dict) -> Optional[Dict]:
    """
    Analyze a single symbol.
    
    Args:
        symbol: Symbol to analyze
        lookback_minutes: Number of minutes to look back
        env: Environment variables
        
    Returns:
        Payload dictionary or None if analysis failed
    """
    try:
        # Normalize symbol
        normalized_symbol = normalize_symbol(symbol)
        
        # Get symbol type and market hours state
        symbol_type = get_symbol_type(normalized_symbol)
        market_hours = trading_hours_state(normalized_symbol)
        
        # Get bars data
        bars = get_bars_for_symbol(normalized_symbol, lookback_minutes)
        
        if not bars or len(bars) < 10:
            print(f"Warning: Insufficient bars for {normalized_symbol}")
            return None
        
        # Set environment variables for this symbol
        env["SYMBOL"] = normalized_symbol
        env["LOOKBACK_MINUTES"] = str(lookback_minutes)
        
        # Run analysis (this would need to be refactored from tracer_bullet.py)
        # For now, we'll create a minimal payload
        payload = _create_minimal_payload(normalized_symbol, bars, symbol_type, market_hours)
        
        return payload
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None


def _create_minimal_payload(symbol: str, bars: List[Dict], symbol_type: str, market_hours: Dict) -> Dict:
    """
    Create a minimal payload for testing.
    
    Args:
        symbol: Symbol string
        bars: Bars data
        symbol_type: Type of symbol
        market_hours: Market hours state
        
    Returns:
        Minimal payload dictionary
    """
    # Calculate basic metrics
    if len(bars) >= 2:
        first_close = bars[0]["close"]
        last_close = bars[-1]["close"]
        price_change_pct = ((last_close - first_close) / first_close) * 100
    else:
        price_change_pct = 0.0
    
    # Generate some mock analysis results
    import random
    random.seed(hash(symbol))
    
    divergence = (random.random() - 0.5) * 2.0  # -1 to 1
    confidence = 0.5 + random.random() * 0.4  # 0.5 to 0.9
    volume_z = (random.random() - 0.5) * 2.0  # -1 to 1
    
    payload = {
        "symbol": symbol,
        "symbol_type": symbol_type,
        "market_hours_state": market_hours["state"],
        "divergence": round(divergence, 3),
        "confidence": round(confidence, 3),
        "volume_z": round(volume_z, 3),
        "price_move_pct": round(price_change_pct, 2),
        "action": "HOLD",
        "summary": f"Analysis for {symbol}",
        "detail": f"Basic analysis completed for {symbol}",
        "timestamp": datetime.now().isoformat()
    }
    
    # Add V2 features
    payload["source_diversity"] = {
        "unique": random.randint(1, 3),
        "top_source_share": round(random.random() * 0.5 + 0.3, 2),
        "counts": {"source1": random.randint(1, 3)},
        "adjustment": round((random.random() - 0.5) * 0.1, 3)
    }
    
    payload["cascade_detector"] = {
        "repetition_ratio": round(random.random(), 2),
        "price_move_pct": abs(price_change_pct),
        "max_volume_z": abs(volume_z),
        "tag": "HYPE_ONLY" if random.random() > 0.7 else "",
        "confidence_delta": -0.03 if random.random() > 0.7 else 0.0
    }
    
    payload["contrarian_viewport"] = "POTENTIAL_CROWD_MISTAKE" if random.random() > 0.8 else ""
    
    # Add V3 features
    payload["timescale_scores"] = {
        "short": {"divergence": round(divergence * 0.8, 3), "price_move_pct": abs(price_change_pct * 0.3), "volume_z": volume_z},
        "mid": {"divergence": round(divergence * 1.1, 3), "price_move_pct": abs(price_change_pct * 0.6), "volume_z": volume_z * 0.8},
        "long": {"divergence": round(divergence * 0.9, 3), "price_move_pct": abs(price_change_pct), "volume_z": volume_z * 0.6},
        "combined_divergence": round(divergence, 3),
        "aligned_horizons": random.randint(1, 3),
        "alignment_flag": random.choice([True, False]),
        "weights": {"short": 0.5, "mid": 0.35, "long": 0.15}
    }
    
    payload["confirmation_checks"] = [
        {"name": "price_vs_narrative", "passed": random.choice([True, False]), "delta": -0.02 if random.random() > 0.7 else 0.0},
        {"name": "volume_support", "passed": random.choice([True, False]), "delta": -0.01 if random.random() > 0.7 else 0.0},
        {"name": "timescale_alignment", "passed": random.choice([True, False]), "delta": -0.02 if random.random() > 0.7 else 0.0}
    ]
    
    payload["confirmation_penalty"] = round(sum(check["delta"] for check in payload["confirmation_checks"]), 3)
    
    payload["position_sizing"] = {
        "confidence": confidence,
        "target_R": round(random.random() * 0.5 + 0.25, 2) if confidence > 0.65 else 0.0,
        "notes": ["linear scaling"] if confidence > 0.65 else ["below floor"],
        "params": {
            "conf_floor": 0.65,
            "conf_cap": 0.85,
            "min_R": 0.25,
            "max_R": 1.00,
            "used_vol_norm": None
        }
    }
    
    return payload


def rank_payloads(payloads: List[Dict]) -> List[Dict]:
    """
    Rank payloads by signal strength.
    
    Args:
        payloads: List of payload dictionaries
        
    Returns:
        Ranked list of payloads
    """
    def sort_key(payload):
        # Primary: absolute divergence (prefer timescale combined if available)
        divergence = abs(payload.get("timescale_scores", {}).get("combined_divergence", payload.get("divergence", 0)))
        
        # Secondary: confidence
        confidence = payload.get("confidence", 0)
        
        # Tertiary: symbol (for deterministic ordering)
        symbol = payload.get("symbol", "")
        
        # Use negative values for reverse=True to get descending order
        return (divergence, confidence, symbol)
    
    # Use stable sort to ensure deterministic ordering
    return sorted(payloads, key=sort_key, reverse=True)


def save_universe_run(payloads: List[Dict], output_dir: str = "universe_runs") -> str:
    """
    Save universe run results.
    
    Args:
        payloads: List of payload dictionaries
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/universe_{timestamp}.json"
    
    # Prepare data for saving
    run_data = {
        "timestamp": datetime.now().isoformat(),
        "total_symbols": len(payloads),
        "payloads": payloads
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(run_data, f, indent=2)
    
    return filename


def _sanitize_version_tag(s: str) -> str:
    if not s:
        return "v31"
    s2 = "".join(ch for ch in s if ch.isalnum()).lower()
    if not s2.startswith("v"):
        s2 = "v" + s2
    return s2


def send_telegram_text(token: str, chat_id: str, text: str) -> bool:
    try:
        import requests
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=15,
        )
        ok = (r.status_code == 200) and (r.json().get("ok") is True)
        if not ok:
            print(f"[Telegram] send failed: {r.status_code} {r.text}")
        return ok
    except Exception as e:
        print(f"[Telegram] send failed: {e}")
        return False


def run_universe_scan(
    config_path: str = None,
    symbols: list = None,
    top_k: int = 5,
    debug: bool = False,
    no_telegram: bool = False,
    version_tag: str = None,
    fail_fast: bool = False,
    max_symbols: int = None
) -> dict:
    load_dotenv()
    if debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
    version = version_tag or "v3.1"
    safe_ver = _sanitize_version_tag(version)
    now = datetime.now(timezone.utc)
    ts_utc_iso = now.replace(microsecond=0).isoformat().replace("+00:00","Z")
    ts_compact = now.strftime("%Y%m%d_%H%M%S")
    # Load universe config or use symbols
    if symbols:
        symlist = [s.strip() for s in symbols]
    else:
        universe = load_universe_config(config_path or "config/universe.yaml")
        symlist = universe["crypto"] + universe["stocks"]
    # Cap symbols if needed
    max_cap = max_symbols or int(os.getenv("TB_UNIVERSE_MAX_SYMBOLS","0") or 0) or None
    if max_cap:
        symlist = symlist[:max_cap]
    lookback_minutes = int(os.getenv("TB_LOOKBACK_OVERRIDE", os.getenv("LOOKBACK_MINUTES", "120")))
    payloads = []
    errors = 0
    processed = 0
    for symbol in symlist:
        try:
            print(f"Analyzing {symbol}...")
            payload = analyze_symbol(symbol, lookback_minutes, os.environ.copy())
            if payload:
                payloads.append(payload)
                print(f"  ✓ {symbol}: div={payload['divergence']:.3f}, conf={payload['confidence']:.3f}")
            else:
                print(f"  ✗ {symbol}: analysis failed")
            processed += 1
        except Exception as e:
            errors += 1
            print(f"[Universe] Symbol {symbol} failed: {e}")
            if fail_fast:
                raise
    if not payloads:
        print("No successful analyses.")
        return {"timestamp_iso": ts_utc_iso, "timestamp_compact": ts_compact, "total_symbols": 0, "universe_file": None, "top_k": top_k, "version": version}
    ranked_payloads = rank_payloads(payloads)
    # Save with version and timestamps in filename and JSON
    output_dir = "universe_runs"
    Path(output_dir).mkdir(exist_ok=True)
    basename = f"universe_{ts_compact}_{safe_ver}.json"
    universe_file = os.path.join(output_dir, basename)
    summary_json = {
        "version": version,
        "timestamp_iso": ts_utc_iso,
        "timestamp_compact": ts_compact,
        "total_symbols": len(ranked_payloads),
        "payloads": ranked_payloads
    }
    with open(universe_file, "w") as f:
        json.dump(summary_json, f, indent=2)
    # Metrics CSV
    write_metrics = os.getenv("TB_UNIVERSE_WRITE_METRICS", "1") == "1"
    files_to_add = [universe_file]
    if write_metrics:
        import csv
        os.makedirs("universe_runs", exist_ok=True)
        metrics_path = os.path.join("universe_runs","metrics.csv")
        header = ["ts","symbol","type","div","combined_div","conf","volz_avg","ts_align","diversity_adj","cascade_tag","target_R"]
        need_header = not os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if need_header:
                w.writeheader()
            for p in payloads:
                ts_scores = p.get("timescale_scores",{}) or {}
                ts_align = ts_scores.get("aligned_horizons")
                combined_div = ts_scores.get("combined_divergence")
                volz_vals = []
                for h in ("short","mid","long"):
                    hv = (ts_scores.get(h) or {}).get("volume_z")
                    if isinstance(hv,(int,float)):
                        volz_vals.append(hv)
                volz_avg = sum(volz_vals)/len(volz_vals) if volz_vals else p.get("volume_z",0.0)
                diversity_adj = (p.get("source_diversity") or {}).get("adjustment",0.0)
                cascade_tag = (p.get("cascade_detector") or {}).get("tag","")
                target_R = (p.get("position_sizing") or {}).get("target_R",0.0)
                w.writerow({
                    "ts": ts_utc_iso,
                    "symbol": p.get("symbol",""),
                    "type": p.get("symbol_type",""),
                    "div": p.get("divergence",0.0),
                    "combined_div": combined_div if combined_div is not None else "",
                    "conf": p.get("confidence",0.0),
                    "volz_avg": volz_avg,
                    "ts_align": ts_align if ts_align is not None else "",
                    "diversity_adj": diversity_adj,
                    "cascade_tag": cascade_tag,
                    "target_R": target_R,
                })
        files_to_add.append(metrics_path)
    # Mirroring/auto-commit/push (same as before, but use universe_file, ts_utc_iso, etc)
    import shutil, subprocess
    mirror_flag = os.getenv("TB_UNIVERSE_MIRROR_TO_RUNS", "0") == "1"
    autoc = os.getenv("TB_UNIVERSE_GIT_AUTOCOMMIT", "0") == "1"
    pushc = os.getenv("TB_UNIVERSE_GIT_PUSH", "0") == "1"
    push_default = os.getenv("TB_UNIVERSE_GIT_PUSH_DEFAULT", "0") == "1"
    if autoc and not pushc and push_default:
        pushc = True
    mirror_path = None
    if mirror_flag:
        try:
            os.makedirs("runs", exist_ok=True)
            mirror_path = os.path.join("runs", os.path.basename(universe_file))
            shutil.copy2(universe_file, mirror_path)
            print(f"[Universe] Mirrored to {mirror_path}")
            files_to_add.append(mirror_path)
        except Exception as e:
            print(f"[Universe] Mirror failed: {e}")
    if autoc:
        try:
            # Safer add: try glob, fallback to explicit
            cmd = 'git add universe_runs/*.json runs/*.json universe_runs/metrics.csv'
            try:
                subprocess.run(cmd, shell=True, check=True)
            except Exception as e:
                print(f"Universe Fallback add failed: {e}")
                try:
                    subprocess.run(["git", "add"] + files_to_add, check=True)
                except Exception as e2:
                    print(f"Universe Explicit add failed: {e2}")
            # Commit message with ISO timestamp and optional Top-N
            allow_empty = os.getenv("TB_UNIVERSE_GIT_ALLOW_EMPTY", "0") == "1"
            append_topn = os.getenv("TB_UNIVERSE_COMMIT_APPEND_TOPN", "0") == "1"
            topn_k_env = os.getenv("TB_UNIVERSE_COMMIT_TOPN_K")
            topn_k = int(topn_k_env) if topn_k_env else top_k
            msg = f"universe: {ts_utc_iso} scanned {len(payloads)} symbols (Top {top_k})"
            if append_topn and payloads:
                def _gap(p):
                    ts = p.get("timescale_scores", {})
                    return ts.get("combined_divergence", p.get("divergence", 0.0))
                ranked = sorted(
                    payloads,
                    key=lambda p: (abs(_gap(p)), p.get("confidence", 0.0), p.get("symbol","")),
                    reverse=True
                )[:topn_k]
                lines = []
                for i, p in enumerate(ranked, 1):
                    gap = _gap(p)
                    conf = p.get("confidence", 0.0)
                    sym = p.get("symbol","")
                    lines.append(f"{i}. {sym} gap {gap:+.2f} conf {conf:.2f}")
                if lines:
                    msg += "\n\nTopN: " + " | ".join(lines)
            commit_cmd = ["git", "commit", "-m", msg]
            if allow_empty:
                commit_cmd.insert(2, "--allow-empty")
            subprocess.run(commit_cmd, check=True)
            print("[Universe] Auto-commit done.")
            if pushc:
                subprocess.run(["git", "push"], check=True)
                print("[Universe] Pushed.")
        except Exception as e:
            print(f"[Universe] Auto-commit failed: {e}")
    # Technical digest (original format)
    digest = format_universe_digest(ranked_payloads, top_k, header_prefix=version, ts=ts_utc_iso)
    if not validate_digest_length(digest):
        digest = truncate_digest(digest)
        print("Warning: Technical digest truncated to fit Telegram limits")
    
    # Human-readable digest (new format)
    human_digest_enabled = os.getenv("TB_HUMAN_DIGEST", "1") != "0"
    human_digest = ""
    if human_digest_enabled:
        summary_data = {
            "timestamp_iso": ts_utc_iso,
            "timestamp_compact": ts_compact,
            "total_symbols": len(ranked_payloads),
            "version": version,
            "payloads": ranked_payloads
        }
        human_digest = render_digest(summary_data)
        print("\nHuman Digest:")
        print("------------")
        print(human_digest)
        print("------------")
    
    # Telegram default ON unless disabled by flag or env
    no_telegram = bool(no_telegram) or (os.getenv("TB_NO_TELEGRAM", "0") == "1")
    if not no_telegram and ranked_payloads:
        # Prefer existing telegram_bot if available; fallback to direct requests
        try:
            # Send human digest if enabled, otherwise send technical digest
            message_to_send = human_digest if human_digest_enabled and human_digest else digest
            success = telegram_bot.send_message(message_to_send)
            print(f"[Telegram] Universe digest sent: {bool(success)}")
        except Exception:
            token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TB_TELEGRAM_BOT_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TB_TELEGRAM_CHAT_ID")
            if token and chat_id:
                # Send human digest if enabled, otherwise send technical digest
                message_to_send = human_digest if human_digest_enabled and human_digest else digest
                sent = send_telegram_text(token, chat_id, message_to_send)
                print(f"[Telegram] Universe digest sent: {bool(sent)}")
            else:
                print("[Telegram] skipped: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    else:
        print("Telegram send disabled")
    return {
        "timestamp_iso": ts_utc_iso,
        "timestamp_compact": ts_compact,
        "total_symbols": len(ranked_payloads),
        "universe_file": universe_file,
        "top_k": top_k,
        "version": version,
        "digest": digest,
        "human_digest": human_digest if human_digest_enabled else "",
        "symbols": symlist
    }


def main():
    parser = argparse.ArgumentParser(description="Scan universe for trading signals")
    parser.add_argument("--config", default="config/universe.yaml", help="Universe config file")
    parser.add_argument("--top", type=int, default=5, help="Number of top signals to show")
    parser.add_argument("--symbols", help="Override symbols (comma-separated)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram sends")
    parser.add_argument("--version-tag", default="v3.1", help="Version tag for this run")
    parser.add_argument("--fail-fast", action="store_true", help="Fail on first error (CI)")
    parser.add_argument("--max-symbols", type=int, default=None, help="Max symbols to scan")
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    run_universe_scan(
        config_path=args.config,
        symbols=symbols,
        top_k=args.top,
        debug=args.debug,
        no_telegram=args.no_telegram,
        version_tag=args.version_tag,
        fail_fast=args.fail_fast,
        max_symbols=args.max_symbols
    )

if __name__ == "__main__":
    main()
