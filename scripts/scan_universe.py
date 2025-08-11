#!/usr/bin/env python3
"""
Universe scanner for multi-asset analysis.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv

from bars_stock import get_bars_stock
from digest_utils import format_universe_digest, validate_digest_length, truncate_digest
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


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Scan universe for trading signals")
    parser.add_argument("--config", default="config/universe.yaml", help="Universe config file")
    parser.add_argument("--top", type=int, default=5, help="Number of top signals to show")
    parser.add_argument("--symbols", help="Override symbols (comma-separated)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram sends")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Set up logging
    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
    
    # Load universe config
    universe = load_universe_config(args.config)
    
    # Override symbols if specified
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
        # Validate symbols
        for symbol in symbols:
            if not get_symbol_type(symbol):
                print(f"Warning: Unknown symbol type for {symbol}")
    else:
        # Use all symbols from config
        symbols = universe["crypto"] + universe["stocks"]
    
    print(f"Scanning {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Get lookback from environment
    lookback_minutes = int(os.getenv("TB_LOOKBACK_OVERRIDE", os.getenv("LOOKBACK_MINUTES", "120")))
    
    # Analyze each symbol
    payloads = []
    for symbol in symbols:
        print(f"Analyzing {symbol}...")
        
        payload = analyze_symbol(symbol, lookback_minutes, os.environ.copy())
        if payload:
            payloads.append(payload)
            print(f"  ✓ {symbol}: div={payload['divergence']:.3f}, conf={payload['confidence']:.3f}")
        else:
            print(f"  ✗ {symbol}: analysis failed")
    
    if not payloads:
        print("No successful analyses.")
        return
    
    # Rank results
    ranked_payloads = rank_payloads(payloads)
    
    # Save results
    output_file = save_universe_run(ranked_payloads)
    print(f"Results saved to {output_file}")
    
    # Auto-commit and mirroring functionality
    import shutil, subprocess
    
    mirror_flag = os.getenv("TB_UNIVERSE_MIRROR_TO_RUNS", "0") == "1"
    autoc = os.getenv("TB_UNIVERSE_GIT_AUTOCOMMIT", "0") == "1"
    pushc = os.getenv("TB_UNIVERSE_GIT_PUSH", "0") == "1"
    
    # Optional mirror into runs/
    mirror_path = None
    if mirror_flag:
        try:
            os.makedirs("runs", exist_ok=True)
            mirror_path = os.path.join("runs", os.path.basename(output_file))
            shutil.copy2(output_file, mirror_path)
            print(f"[Universe] Mirrored to {mirror_path}")
        except Exception as e:
            print(f"[Universe] Mirror failed: {e}")
    
    # Optional git auto-commit
    if autoc:
        try:
            files_to_add = [output_file]
            if mirror_path:
                files_to_add.append(mirror_path)
            # Stage files
            subprocess.run(["git", "add"] + files_to_add, check=True)
            # Commit message
            total = len(payloads)  # we have payloads list in scope
            msg = f"universe: {datetime.now().isoformat()} scanned {total} symbols (Top {args.top})"
            # Commit
            subprocess.run(["git", "commit", "-m", msg], check=True)
            print("[Universe] Auto-commit done.")
            # Optional push
            if pushc:
                subprocess.run(["git", "push"], check=True)
                print("[Universe] Pushed.")
        except Exception as e:
            print(f"[Universe] Auto-commit failed: {e}")
    
    # Generate digest
    digest = format_universe_digest(ranked_payloads, args.top)
    
    # Validate digest length
    if not validate_digest_length(digest):
        digest = truncate_digest(digest)
        print("Warning: Digest truncated to fit Telegram limits")
    
    # Print digest
    print("\n" + "="*50)
    print("UNIVERSE DIGEST")
    print("="*50)
    print(digest)
    print("="*50)
    
    # Send to Telegram if enabled
    if not args.no_telegram and not os.getenv("TB_NO_TELEGRAM"):
        try:
            success = telegram_bot.send_message(digest)
            if success:
                print("✓ Digest sent to Telegram")
            else:
                print("✗ Failed to send digest to Telegram")
        except Exception as e:
            print(f"✗ Error sending to Telegram: {e}")
    else:
        print("Telegram send disabled")


if __name__ == "__main__":
    main()
