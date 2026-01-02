from __future__ import annotations

import json
import logging
from io import StringIO
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from quant_engine.utils.logger import (
    ContextFilter,
    JsonFormatter,
    _debug_module_matches,
    get_logger,
    init_logging,
    log_data_integrity,
    log_debug,
    log_info,
    safe_jsonable,
)


class Color(Enum):
    RED = "red"


@dataclass
class Sample:
    path: Path
    created: datetime
    color: Color


def test_safe_jsonable_handles_common_types() -> None:
    payload = {
        "path": Path("foo/bar"),
        "created": datetime(2020, 1, 1, 0, 0, 0),
        "enum": Color.RED,
        "sample": Sample(Path("x/y"), datetime(2021, 1, 2, 3, 4, 5), Color.RED),
        "exc": ValueError("boom"),
        "tuple": (1, 2),
        "set": {3, 4},
    }

    out = safe_jsonable(payload)
    json.dumps(out)
    assert out["path"] == "foo/bar"
    assert "2020" in out["created"]


def test_debug_module_matching() -> None:
    assert _debug_module_matches("quant_engine.strategy.engine", "strategy")
    assert _debug_module_matches("quant_engine.strategy.engine", "quant_engine.strategy")
    assert _debug_module_matches("strategy.engine", "strategy")
    assert not _debug_module_matches("quant_engine.models", "strategy")


def test_init_logging_dictconfig_applied(tmp_path: Path) -> None:
    config = {
        "active_profile": "default",
        "profiles": {
            "default": {
                "level": "DEBUG",
                "debug": {"enabled": True, "modules": ["strategy"]},
                "handlers": {"console": {"enabled": True, "level": "DEBUG"}},
                "format": {"json": True, "timestamp_utc": True},
            }
        },
    }
    config_path = tmp_path / "logging.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    init_logging(config_path=str(config_path))
    logger = get_logger("quant_engine.test")

    assert logging.getLogger().level == logging.DEBUG
    assert logger.getEffectiveLevel() == logging.DEBUG

    root_handlers = logging.getLogger().handlers
    assert root_handlers
    assert any(isinstance(h.formatter, JsonFormatter) for h in root_handlers)
    assert any(any(isinstance(f, ContextFilter) for f in h.filters) for h in root_handlers)


def test_init_logging_file_output_with_context_and_category(tmp_path: Path) -> None:
    log_path = tmp_path / "runs" / "{run_id}" / "logs" / "{mode}.jsonl"
    config = {
        "active_profile": "default",
        "profiles": {
            "default": {
                "level": "INFO",
                "debug": {"enabled": False, "modules": []},
                "handlers": {
                    "console": {"enabled": False},
                    "file": {"enabled": True, "level": "INFO", "path": str(log_path)},
                },
                "format": {"json": True},
            },
            "backtest": {
                "level": "INFO",
                "debug": {"enabled": False, "modules": []},
                "handlers": {
                    "console": {"enabled": False},
                    "file": {"enabled": True, "level": "INFO", "path": str(log_path)},
                },
                "format": {"json": True},
            }
        },
        
    }
    config_path = tmp_path / "logging.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    init_logging(config_path=str(config_path), run_id="run123", mode="backtest")
    logger = get_logger("quant_engine.test")
    log_data_integrity(logger, "data.gap_detected", symbol="BTCUSDT", gap_type="missing")

    resolved_path = tmp_path / "runs" / "run123" / "logs" / "backtest.jsonl"
    lines = resolved_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])

    assert payload["event"] == "data.gap_detected"
    assert payload["category"] == "data_integrity"
    assert payload["context"]["run_id"] == "run123"
    assert payload["context"]["mode"] == "backtest"
    assert "category" not in payload.get("context", {})
    assert {"ts", "ts_ms", "level", "logger", "event", "module", "msg"}.issubset(payload.keys())

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def test_log_debug_gating_with_monkeypatch(monkeypatch) -> None:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(ContextFilter())

    logger_match = logging.getLogger("quant_engine.data.cache")
    logger_match.handlers = []
    logger_match.addHandler(handler)
    logger_match.setLevel(logging.DEBUG)
    logger_match.propagate = False

    logger_miss = logging.getLogger("quant_engine.strategy.engine")
    logger_miss.handlers = []
    logger_miss.addHandler(handler)
    logger_miss.setLevel(logging.DEBUG)
    logger_miss.propagate = False

    monkeypatch.setattr("quant_engine.utils.logger._CONFIGURED", True)
    monkeypatch.setattr("quant_engine.utils.logger._RUN_ID", "rid")
    monkeypatch.setattr("quant_engine.utils.logger._MODE", "mode")
    monkeypatch.setattr("quant_engine.utils.logger._DEBUG_ENABLED", True)
    monkeypatch.setattr("quant_engine.utils.logger._DEBUG_MODULES", {"quant_engine.data"})

    log_debug(logger_miss, "smoke.debug", detail="skip")
    log_debug(logger_match, "smoke.debug", detail="ok")
    log_info(logger_match, "smoke.start", sample=1)

    lines = stream.getvalue().strip().splitlines()
    assert len(lines) == 2
    debug_payload = json.loads(lines[0])
    info_payload = json.loads(lines[1])

    assert debug_payload["event"] == "smoke.debug"
    assert debug_payload["context"]["run_id"] == "rid"
    assert debug_payload["context"]["mode"] == "mode"
    assert info_payload["event"] == "smoke.start"
