

from __future__ import annotations

from typing import Any, Mapping, Iterable

from ingestion.contracts.tick import IngestionTick, Domain
from ingestion.contracts.normalize import Normalizer


class GenericOptionChainNormalizer(Normalizer):
    """
    Generic option chain normalizer.
    """
    
    symbol: str
    domain: Domain = "option_chain"

    def __init__(self, symbol: str):
        self.symbol = symbol

    def normalize(
        self,
        *,
        raw: Mapping[str, Any],
    ) -> IngestionTick:
        """
        Normalize a raw option chain payload into an IngestionTick.
        Expected (minimal) raw fields:
            - timestamp / ts / snapshot_time   (seconds or ms)
            - symbol / underlying
        """

        # --- timestamp extraction (best-effort, ingestion-level only) ---
        if "timestamp" in raw:
            ts = float(raw["timestamp"])
        elif "snapshot_time" in raw:
            ts = float(raw["snapshot_time"])
        elif "ts" in raw:
            ts = float(raw["ts"])
        else:
            raise ValueError("Option chain payload missing timestamp field")

        # heuristic: milliseconds -> seconds
        if ts > 1e12:
            ts = ts / 1000.0

        # --- underlying symbol association ---
        symbol = (
            raw.get("symbol")
            or raw.get("underlying")
            or raw.get("asset")
        )

        if symbol is None:
            raise ValueError("Option chain payload missing underlying symbol")

        return IngestionTick(
            domain=self.domain,
            symbol=self.symbol,
            timestamp=ts,
            data_ts=ts,  # arrival time not guaranteed; default to event time
            payload=dict(raw),
        )