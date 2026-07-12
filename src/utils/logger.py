"""Loguru-based logging configuration for ACN.

Usage in any module:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Something happened")
    logger.debug("Debug detail: {}", value)

Application startup should call `configure_logging(cfg)` once.
"""

import os
import sys
from typing import Optional

from loguru import logger

# Default format used by all sinks
_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan> | "
    "{message}"
)


def configure_logging(config: Optional[dict] = None) -> None:
    """
    Configure loguru sinks once at application startup.

    Args:
        config: Dict with optional keys:
            - level (str): minimum global log level, default "INFO"
            - file_level (str): file sink level, default "DEBUG"
            - console_level (str): stderr sink level, default "WARNING"
            - log_dir (str): directory for log files, default "logs"
            - log_filename (str): log file name, default "acn.log"
            - rotation (str): loguru rotation rule, default "5 MB"
            - retention (str): loguru retention rule, default "1 week"
            - format (str): log format string, default built-in format
    """
    if config is None:
        config = {}

    # Remove default sink to avoid duplicate output
    logger.remove()

    file_level = config.get("file_level", "DEBUG")
    console_level = config.get("console_level", "WARNING")
    log_dir = config.get("log_dir", "logs")
    log_filename = config.get("log_filename", "acn.log")
    rotation = config.get("rotation", "5 MB")
    retention = config.get("retention", "1 week")
    fmt = config.get("format", _DEFAULT_FORMAT)

    # Console sink — WARNING by default to keep console quiet
    logger.add(
        sys.stderr,
        level=console_level,
        format=fmt,
        filter=lambda record: record["level"].no >= logger.level(console_level).no,
    )

    # File sink — DEBUG by default for detailed logs
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    logger.add(
        log_path,
        level=file_level,
        format=fmt,
        rotation=rotation,
        retention=retention,
        enqueue=True,
    )

    logger.info("Logging configured: console={} file={}", console_level, log_path)


def get_logger(name: str):
    """
    Get a logger bound to a module name.

    With loguru this returns the shared logger (the name is included
    automatically via the {name} format field). The name argument is
    kept for API symmetry with Python's logging.getLogger().

    Args:
        name: Typically ``__name__`` from the calling module.
    """
    return logger
