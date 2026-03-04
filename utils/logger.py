"""
logger.py — Centralised logging for AI Silent Disease Predictor.

Every module obtains its logger via ``get_logger(__name__)``.
Logs are emitted to both the console and a rotating log file.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from config.settings import LOG_DIR, LOG_FILE, LOG_FORMAT, LOG_LEVEL

# ---------------------------------------------------------------------------
# Ensure log directory exists
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Module-level flag to prevent duplicate handler attachment
# ---------------------------------------------------------------------------
_CONFIGURED = False


def _configure_root() -> None:
    """Attach handlers to the root logger once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(console)

    # File handler (rotates at 5 MB, keeps 3 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Configures handlers on first call.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    _configure_root()
    return logging.getLogger(name)
