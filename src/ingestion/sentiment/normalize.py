from __future__ import annotations
from typing import Any, Mapping
from ingestion.contracts.tick import IngestionTick, Domain
from ingestion.contracts.normalize import Normalizer


class GenericSentimentNormalizer(Normalizer):
    """
    Generic sentiment normalizer.
    """
    symbol: str
    domain: Domain = "sentiment"
    
    def __init__(self, symbol: str):
        self.symbol = symbol

    def normalize(
        self,
        *,
        raw: Mapping[str, Any],
    ) -> IngestionTick:
        """
        Normalize a raw sentiment payload into an IngestionTick.
        Expected (minimal) raw fields:
            - timestamp / published_at / ts  (seconds or ms)
        Optional:
            - symbol / asset / ticker
            - source / vendor / category
            - text / score / embedding ref
        The payload is passed through largely untouched.
        """

        # --- timestamp extraction (best-effort, ingestion-level only) ---
        if "timestamp" in raw:
            ts = float(raw["timestamp"])
        elif "published_at" in raw:
            ts = float(raw["published_at"])
        elif "ts" in raw:
            ts = float(raw["ts"])
        else:
            raise ValueError("Sentiment payload missing timestamp field")

        # heuristic: milliseconds -> seconds
        if ts > 1e12:
            ts = ts / 1000.0

        # --- symbol association (optional) ---
        symbol = (
            raw.get("symbol")
            or raw.get("asset")
            or raw.get("ticker")
            or "GLOBAL"
        )

        return IngestionTick(
            domain=self.domain,
            symbol=self.symbol,
            timestamp=ts,
            data_ts=ts,  # arrival time not guaranteed; default to event time
            payload=dict(raw),
        )