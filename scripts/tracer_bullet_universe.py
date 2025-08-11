#!/usr/bin/env python3
"""
Canonical multi-asset universe entrypoint for Tracer Bullet.
"""
import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.scan_universe import run_universe_scan

def main():
    parser = argparse.ArgumentParser(description="Tracer Bullet Universe Scanner")
    parser.add_argument("--config", default="config/universe.yaml", help="Universe config file")
    parser.add_argument("--symbols", help="Override symbols (comma-separated)")
    parser.add_argument("--top", type=int, default=5, help="Number of top signals to show")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram sends")
    parser.add_argument("--version-tag", default="v3.1", help="Version tag for this run")
    parser.add_argument("--fail-fast", action="store_true", help="Fail on first error (CI)")
    parser.add_argument("--max-symbols", type=int, default=None, help="Max symbols to scan")
    parser.add_argument("--no-human-digest", action="store_true", help="Disable human-readable digest")
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    
    # Set human digest environment variable based on CLI flag
    if args.no_human_digest:
        os.environ["TB_HUMAN_DIGEST"] = "0"
    
    summary = run_universe_scan(
        config_path=args.config,
        symbols=symbols,
        top_k=args.top,
        debug=args.debug,
        no_telegram=args.no_telegram,
        version_tag=args.version_tag,
        fail_fast=args.fail_fast,
        max_symbols=args.max_symbols
    )
    print(f"[{args.version_tag}] [TracerBullet Universe] ts={summary.get('timestamp_iso')} top={summary['top_k']}")
    print(f"[{summary.get('version','v3.1')}] Universe scanned: {summary['total_symbols']} symbols; Top {summary['top_k']} ranked. File: {summary['universe_file']}")
    # Print technical digest summary if available
    if summary.get("digest") and not summary.get("human_digest"):
        print("\nTop-N Digest:")
        print("\n".join(summary["digest"].splitlines()[:args.top+2]))  # header + N lines

if __name__ == "__main__":
    main()
