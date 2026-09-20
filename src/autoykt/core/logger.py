"""Idempotent console and JSONL application logging."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys


_LOGGER_NAME = "autoykt"


class JsonFormatter(logging.Formatter):
    """Serialize standard records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logger(
    level: str = "INFO",
    log_dir: str | Path = "storage/logs",
) -> logging.Logger:
    """Configure and return the application logger exactly once."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    for handler in tuple(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(module)-18s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(console)

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    file_handler = RotatingFileHandler(
        directory / f"{day}.jsonl",
        encoding="utf-8",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
    return logger
