import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

# local imports after sys.path tweak for repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preflight import health as preflight_health  # noqa: E402
from logging_utils import setup_logging  # noqa: E402


def _load_env() -> None:
    try:
        env_path = find_dotenv(usecwd=True)
        load_dotenv(env_path or None)
    except Exception:
        try:
            load_dotenv()
        except Exception:
            pass


def main() -> int:
    """
    CLI entrypoint for Tracer Bullet.

    Overrides precedence (highest first):
      1) CLI flags (--symbol, --lookback, --no-telegram, --debug)
      2) Environment variables (SYMBOL, LOOKBACK_MINUTES, TB_NO_TELEGRAM, LOG_LEVEL)
      3) Defaults in config.py

    The TB_* flags are intended for automation/testing; avoid hardcoding.
    """
    parser = argparse.ArgumentParser(description="Tracer Bullet runner")
    parser.add_argument("--symbol", dest="symbol", default=None, help="Override symbol (e.g., BTC/USD)")
    parser.add_argument("--lookback", dest="lookback", type=int, default=None, help="Lookback minutes")
    parser.add_argument("--no-telegram", dest="no_telegram", action="store_true", help="Disable Telegram push for this run")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Set DEBUG logging")
    parser.add_argument("--health", dest="health", action="store_true", help="Run preflight health checks and exit")
    args = parser.parse_args()

    _load_env()

    if args.health:
        return preflight_health(verbose=True)

    # Configure logging early
    setup_logging(debug=args.debug)

    # Apply environment overrides expected by the app
    # CLI has highest precedence
    if args.symbol:
        os.environ["SYMBOL"] = args.symbol
    if args.lookback is not None:
        os.environ["LOOKBACK_MINUTES"] = str(args.lookback)
    # Next, TB_* overrides if provided and not set by CLI
    tb_sym = os.getenv("TB_SYMBOL_OVERRIDE", "").strip()
    if not args.symbol and tb_sym:
        os.environ["SYMBOL"] = tb_sym
    tb_look = os.getenv("TB_LOOKBACK_OVERRIDE", "").strip()
    if args.lookback is None and tb_look.isdigit():
        os.environ["LOOKBACK_MINUTES"] = tb_look
    if args.no_telegram:
        os.environ["TB_NO_TELEGRAM"] = "1"
    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"

    # Lazily import after env overrides so consumers pick them up
    from tracer_bullet import main as tracer_main  # type: ignore

    # Some components respect TB_*; to keep behavior simple we rely on tracer_bullet
    # to check TB_NO_TELEGRAM env (we add a small shim if not present).
    return tracer_main() or 0


if __name__ == "__main__":
    sys.exit(main())


