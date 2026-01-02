# Logging Usage Spec

## Purpose
This document defines how logging is initialized and used across Quant Engine.

## Initialization (entrypoints)
- `apps/run_backtest.py`: calls `init_logging(run_id=_RUN_ID)` before importing runtime/ingestion modules.
- `apps/run_realtime.py`: calls `init_logging(run_id=_RUN_ID)` before importing runtime/ingestion modules.
- `scripts/logging_smoke.py`: demo script that loads `configs/logging.json` and initializes logging.

If you add a new process entrypoint under `apps/`, call `init_logging(...)` once at the top of the file before most imports.
Pass `run_id` for context injection and `mode` (profile name) if you want a non-default profile.

## Logger acquisition
- Use `from quant_engine.utils.logger import get_logger`.
- Call `get_logger(__name__)` (or a stable explicit name) in each module or class.
- Do not call `logging.getLogger(...)` directly outside `quant_engine/utils/logger.py`.

## Emission policy
Use helper functions so context is JSON-safe and consistently shaped:
- `log_info(logger, "event.name", **context)`
- `log_warn(logger, "event.name", **context)`
- `log_exception(logger, "event.name", **context)` inside an exception handler
- `log_debug(logger, "event.name", **context)` for gated debug
- Domain helpers: `log_data_integrity`, `log_heartbeat`, `log_decision`, `log_execution`, `log_portfolio`

Avoid calling `logger.info(...)` with manual `extra={...}` unless you are in a low-level adapter (e.g., `runtime/log_router.py`).

## Debug gating logic
Debug output is controlled by the active profile:
- `debug.enabled`: must be true for `log_debug` to emit.
- `debug.modules`: optional list of allowed module prefixes.

Module matching behavior:
- Matches if `logger.name == module`
- Matches if `logger.name` starts with `module + "."`
- Matches if any dot-separated part of `logger.name` equals `module`

## Output format and file paths
`configs/logging.json` defines profiles:
- `default`: JSON format, console enabled, file disabled.
- `monitor`: JSON format, console enabled, file enabled.
- `debug_manual`, `debug_data`, `debug_strat`: DEBUG level, console enabled, file disabled, debug gating enabled.

Default file template (when enabled):
`artifacts/runs/{run_id}/logs/{mode}.jsonl`

`init_logging(run_id=..., mode=...)` injects `run_id` and `mode` into log context fields.

## Do all layers get the same logger?
Not exactly. Python logging keeps a global registry keyed by logger name.
`get_logger(name)` returns the logger object for that name; different names mean different Logger objects.
However, they typically share the same root handlers and formatters configured by `init_logging`
unless a module adds per-logger handlers or overrides propagation.
So layers share the same configuration and outputs, but loggers are distinct by name and identity.
