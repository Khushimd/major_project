"""
Logging utilities for the Melanoma CDSS.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import config


def setup_logger(
    name: str = "melanoma_cdss",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a named logger with console and optional file handlers.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional log file name inside LOGS_DIR.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    filename = log_file or f"{name}.log"
    file_path: Path = config.LOGS_DIR / filename
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Default application logger
logger = setup_logger()
