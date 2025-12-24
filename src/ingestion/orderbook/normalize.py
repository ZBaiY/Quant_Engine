

from __future__ import annotations

from typing import Any, Mapping, Dict, List, Tuple

from ingestion.contracts.tick import IngestionTick, Domain
from ingestion.contracts.normalize import Normalizer


class BinanceOrderbookNormalizer(Normalizer):
    """
    Normalize Binance orderbook payloads into IngestionTick.
    """

    symbol: str
    domain: Domain = "orderbook"

    def __init__(self, symbol: str):
        self.symbol = symbol

    def normalize(
        self,
        *,
        raw: Mapping[str, Any],
    ) -> IngestionTick:
        """
        Normalize a single orderbook payload into an IngestionTick.
        """

        # --- detect payload shape ---
        # WS: { "e": "depthUpdate", "E": ..., "s": "BTCUSDT", "U": ..., "u": ..., "b": [...], "a": [...] }
        # REST: { "lastUpdateId": ..., "bids": [...], "asks": [...] }

        if "b" in raw and "a" in raw:  # WebSocket depth update
            bids = raw["b"]
            asks = raw["a"]
            event_ts = float(raw.get("E", 0.0)) / 1000.0
            sym = self.symbol or raw.get("s")
        elif "bids" in raw and "asks" in raw:  # REST snapshot
            bids = raw["bids"]
            asks = raw["asks"]
            event_ts = float(raw.get("lastUpdateId", 0.0))
            sym = self.symbol
        else:
            raise ValueError("Unsupported orderbook payload format")

        if sym is None:
            raise ValueError("Symbol must be provided or present in raw payload")

        # --- normalize levels ---
        def _levels(rows) -> List[Tuple[float, float]]:
            out: List[Tuple[float, float]] = []
            for price, qty in rows:
                out.append((float(price), float(qty)))
            return out

        payload = {
            "bids": _levels(bids),
            "asks": _levels(asks),
        }

        # data_ts: arrival time approximated by event time if not provided
        data_ts = event_ts

        return IngestionTick(
            domain=self.domain,
            symbol=self.symbol,
            timestamp=event_ts,
            data_ts=data_ts,
            payload=payload,
        )