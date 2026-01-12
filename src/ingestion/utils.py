from __future__ import annotations

from typing import Any

from quant_engine.utils.logger import log_warn


def resolve_poll_interval_ms(
    logger: Any,
    *,
    poll_interval_ms: int | None,
    interval_ms: int | None,
    log_context: dict[str, Any],
) -> int | None:
    if poll_interval_ms is None or interval_ms is None:
        return poll_interval_ms
    if int(poll_interval_ms) == int(interval_ms):
        return int(poll_interval_ms)
    log_warn(
        logger,
        "ingestion.poll_interval_override",
        **log_context,
        interval_ms=int(interval_ms),
        poll_interval_ms=int(poll_interval_ms),
    )
    return int(interval_ms)
