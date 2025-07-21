"""
Logging configuration for the Turboprop semantic code search system.

This module sets up structured logging throughout the application using the
configuration from config.py.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from .config import config


def setup_logging(
    logger_name: str = "turboprop",
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        logger_name: Name of the logger
        log_level: Override log level from config
        log_file: Override log file from config

    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(logger_name)

    # Don't configure if already configured
    if logger.handlers:
        return logger

    # Set log level
    level = log_level or config.logging.LOG_LEVEL
    logger.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(config.logging.LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    file_path = log_file or config.logging.LOG_FILE
    if file_path:
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=config.logging.LOG_MAX_SIZE,
            backupCount=config.logging.LOG_BACKUP_COUNT,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent duplicate logs
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return setup_logging(f"turboprop.{name}")


# Create module-level logger
logger = get_logger(__name__)
