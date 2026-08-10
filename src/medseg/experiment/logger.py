"""Logging helpers for experiments."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logger(log_directory: Path | None = None, logger_name: str = "medseg") -> logging.Logger:
    """Configure a console logger and an optional file logger."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_directory is not None:
        log_directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_directory / "run.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
