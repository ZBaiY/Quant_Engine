from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Literal

Domain = Literal[
    "ohlcv",
    "orderbook",
    "trade",
    "option_chain",
    "iv_surface",
    "sentiment",
]


@dataclass(frozen=True)
class IngestionTick:
    """
    Canonical ingestion tick.

    This is the ONLY object allowed to cross the boundary:
        Ingestion -> Driver -> Engine -> DataHandler

    Semantics:
        - `timestamp`      : event timestamp from source / logical event time (float, seconds)
        - `data_ts`        : ingestion arrival timestamp (float, seconds)
        - `domain`         : data domain identifier (e.g. 'ohlcv', 'orderbook')
        - `symbol`         : instrument symbol (e.g. 'BTCUSDT')
        - `payload`        : normalized domain-specific data
    """

    timestamp: float
    data_ts: float
    domain: Domain
    symbol: str
    payload: Mapping[str, Any]


def normalize_tick(
    *,
    timestamp: float,
    domain: Domain,
    symbol: str,
    payload: Mapping[str, Any],
    data_ts: float | None = None,
) -> IngestionTick:
    """
    Normalize raw ingestion output into a canonical IngestionTick.

    Rules:
        - timestamp is ALWAYS provided by ingestion controller
        - data_ts defaults to timestamp if source timestamp is missing
        - no mutation, no enrichment, no inference
    """
    ts = float(timestamp)
    event_ts = float(data_ts) if data_ts is not None else ts

    return IngestionTick(
        timestamp=ts,
        data_ts=event_ts,
        domain=domain,
        symbol=str(symbol),
        payload=payload,
    )
