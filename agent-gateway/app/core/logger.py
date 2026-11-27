
import json
import sys
from pathlib import Path

from loguru import logger

from app.core.config import settings


def json_formatter(record):
    extra = record["extra"]

    log_data = {

        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "request_id": extra.get("request_id"),
        "trace_id": extra.get("trace_id"),
        "span_id": extra.get("span_id"),
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "exception": (
            str(record["exception"])
            if record["exception"]
            else None
        )
    }

    return json.dumps(log_data, ensure_ascii=False)


def setup_logger():

    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stdout,
        format="{message}",
        level=settings.LOG_LEVEL,
        colorize=False,
        enqueue=True
    )

    logger.add(
        str(log_dir / "agent-gateway.log"),
        format="{message}",
        level=settings.LOG_LEVEL,
        rotation="100 MB",
        retention="30 days",
        encoding="utf-8",
        enqueue=True,
        serialize=False,
        colorize=False
    )

    return logger


app_logger = setup_logger()
