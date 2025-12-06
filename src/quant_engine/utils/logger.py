import logging
import json
from logging import Logger
from functools import lru_cache
from datetime import datetime
import os
import pathlib

# Load logging config
def _load_logging_config():
    config_path = pathlib.Path(__file__).resolve().parents[2] / "configs" / "logging.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "level": "INFO",
        "debug": {"enabled": False, "modules": []},
        "handlers": {"console": {"enabled": True}},
        "format": {"json": True, "timestamp_utc": True}
    }

_LOG_CFG = _load_logging_config()
DEBUG_ENABLED = _LOG_CFG.get("debug", {}).get("enabled", False)
DEBUG_MODULES = set(_LOG_CFG.get("debug", {}).get("modules", []))
GLOBAL_LEVEL = getattr(logging, _LOG_CFG.get("level", "INFO").upper(), logging.INFO)


class ContextFilter(logging.Filter):
    """Guarantees record.context always exists (prevents type warnings)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "context"):
            record.context = None
        return True
    
class JsonFormatter(logging.Formatter):
    """Structured JSON formatter for deterministic, parseable logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
        }

        if record.context: # type: ignore
            payload["context"] = record.context # pyright: ignore[reportAttributeAccessIssue]

        return json.dumps(payload, ensure_ascii=False)



@lru_cache(None)
def get_logger(name: str = "quant_engine", level: int = logging.INFO) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(GLOBAL_LEVEL)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.addFilter(ContextFilter())
    handler.setFormatter(JsonFormatter())

    logger.addHandler(handler)
    return logger


def log_debug(logger: Logger, msg: str, **context):
    module_name = logger.name.split(".")[0]
    if not DEBUG_ENABLED:
        return
    if DEBUG_MODULES and (module_name not in DEBUG_MODULES):
        return
    logger.debug(msg, extra={"context": context})


def log_info(logger: Logger, msg: str, **context):
    logger.info(msg, extra={"context": context})


def log_warn(logger: Logger, msg: str, **context):
    logger.warning(msg, extra={"context": context})


def log_error(logger: Logger, msg: str, **context):
    logger.error(msg, extra={"context": context})