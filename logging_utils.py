import logging
import os


def setup_logging(debug: bool = False) -> None:
    """
    Configure root logging with ISO-8601 timestamps and sensible library noise suppression.

    - debug=False → INFO level
    - debug=True  → DEBUG level
    - Silence 'requests' and 'urllib3' below WARNING
    """
    level = logging.DEBUG if debug else logging.INFO
    # Respect explicit LOG_LEVEL if provided (e.g., LOG_LEVEL=DEBUG)
    env_level = os.getenv("LOG_LEVEL", "").upper()
    if env_level in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        level = getattr(logging, env_level)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    # Quiet noisy libraries
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


