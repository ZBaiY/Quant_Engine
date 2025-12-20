from __future__ import annotations

from typing import Protocol, Mapping, Any, runtime_checkable


@runtime_checkable
class Snapshot(Protocol):
    """
    Immutable runtime snapshot contract.

    A Snapshot represents a frozen view of domain data at engine clock `timestamp`,
    derived from underlying data timestamp `data_ts`.

    Invariants:
    - timestamp >= data_ts (anti-lookahead)
    - latency == timestamp - data_ts
    - to_dict() returns pure-python serializable objects
    - schema_version is used for tolerant evolution of snapshot fields
    """

    # --- timing ---
    timestamp: float          # engine clock timestamp
    symbol: str           # associated symbol

    data_ts: float     # data-origin timestamp
    latency: float     # timestamp - data_ts
    # --- identity ---
    domain: str        # "ohlcv" | "orderbook" | "option_chain" | "iv_surface" | ...
    schema_version: int  # for forward/backward compatibility

    def to_dict(self) -> Mapping[str, Any]:
        """
        Return a pure-python, serialization-safe representation.

        Must NOT return pandas / numpy objects.
        Must NOT expose internal mutable references.
        """
        ...