"""Unified logging: console + file with structured JSON for answer records."""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON lines for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_data", None)
        if extra is not None:
            log_data["data"] = extra
        return json.dumps(log_data, ensure_ascii=False)


def setup_logger(level: str = "INFO", log_dir: str = "storage/logs") -> logging.Logger: #初始化log记录器
    """Set up the application logger with console and file handlers."""
    log_path = Path(log_dir)                                                    #创建logs目录，如果不存在
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("auto_answer")                                   #创建标准logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler - human readable
    console = logging.StreamHandler(sys.stdout)                                 #handler使log发布到console
    console.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(module)-15s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    # File handler - JSON lines
    today = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(
        log_path / f"{today}.jsonl", encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())                                  #handler使log同时发布到每天的JSON文件
    logger.addHandler(file_handler)

    return logger


def log_answer_record(logger: logging.Logger, data: dict[str, Any]) -> None:    #手动log，带extra_data
    """Log a structured answer record (question, answer, result, timing)."""    #用于记录具体题目及回答
    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn="",
        lno=0,
        msg=f"Answer record: {data.get('question_text', '')[:50]}...",
        args=(),
        exc_info=None,
    )
    record.extra_data = data  # type: ignore[attr-defined]
    logger.handle(record)