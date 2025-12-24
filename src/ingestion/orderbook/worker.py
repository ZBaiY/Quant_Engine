from __future__ import annotations

import asyncio
from typing import Callable

from ingestion.contracts.tick import IngestionTick
from ingestion.contracts.worker import IngestWorker
from ingestion.orderbook.normalize import BinanceOrderbookNormalizer
from ingestion.orderbook.source import OrderbookFileSource, OrderbookRESTSource, OrderbookWebSocketSource

class OrderbookWorker(IngestWorker):
    """
    Orderbook ingestion worker.
    This worker is source-agnostic and supports:
        - WebSocket streaming (AsyncSource)
        - REST polling / replay (Source)

    Responsibilities:
        raw -> normalize -> emit tick
    """

    def __init__(
        self,
        *,
        normalizer: BinanceOrderbookNormalizer,
        source: OrderbookFileSource | OrderbookRESTSource | OrderbookWebSocketSource,
        symbol: str,
        poll_interval: float | None = None,
    ):
        self._normalizer = normalizer
        self._source = source
        self._symbol = symbol
        self._poll_interval = poll_interval

    async def run(self, emit: Callable[[IngestionTick], None]) -> None:
        # --- async source (e.g. WebSocket depth stream) ---
        if hasattr(self._source, "__aiter__"):
            async for raw in self._source:  # type: ignore
                tick = self._normalize(raw)
                emit(tick)

        # --- sync source (e.g. REST snapshot / backtest replay) ---
        else:
            for raw in self._source:  # type: ignore
                tick = self._normalize(raw)
                emit(tick)
                if self._poll_interval is not None:
                    await asyncio.sleep(self._poll_interval)

    def _normalize(self, raw: dict) -> IngestionTick:
        return self._normalizer.normalize(raw=raw)
