from __future__ import annotations

from typing import Any, Callable, Iterable

from quant_engine.data.ohlcv.realtime import RealTimeDataHandler
from quant_engine.strategy.engine import StrategyEngine


class LiveEngine:
    """
    Unified Live Trading Engine (realtime driver).

    Semantics:
        - Data ingestion runs independently (feed / websocket / poller).
        - Strategy execution is EVENT-DRIVEN by the primary clock
          (typically OHLCV bar close).
        - Engine NEVER fetches data itself.
    """

    def __init__(
        self,
        *,
        engine: StrategyEngine,
        ohlcv_handler: RealTimeDataHandler,
        feed: Iterable[dict] | None = None,
        on_snapshot: Callable[[Any], None] | None = None,
    ):
        """
        Parameters
        ----------
        engine:
            Assembled StrategyEngine (already preloaded + warmed up).
        ohlcv_handler:
            Primary OHLCV handler acting as the realtime clock source.
        feed:
            Iterable / generator yielding OHLCV bars, e.g.
            {
                "timestamp": float,
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
            }
        on_snapshot:
            Optional callback invoked after each engine.step().
        """
        self.engine = engine
        self.ohlcv_handler = ohlcv_handler
        self.feed = feed
        self._on_snapshot = on_snapshot or self._default_on_snapshot

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        if self.feed is None:
            raise ValueError("LiveEngine requires a feed iterable.")

        # ---- lifecycle guard ----
        if not getattr(self.engine, "_warmup_done", False):
            raise RuntimeError(
                "LiveEngine.run() requires engine.preload_data() "
                "and engine.warmup_features() to be called first"
            )
        # -------------------------

        for bar in self.feed:
            self._ingest_bar(bar)

            # Primary clock: OHLCV bar close timestamp
            ts = self._extract_timestamp(bar)

            snapshot = self.engine.step(ts=ts)
            self._on_snapshot(snapshot)

    # ------------------------------------------------------------------
    # Hooks (override-friendly)
    # ------------------------------------------------------------------

    def _ingest_bar(self, bar: dict) -> None:
        """
        Push one OHLCV bar into the realtime handler.
        """
        self.ohlcv_handler.on_new_tick(bar)

    def _extract_timestamp(self, bar: dict) -> float:
        """
        Extract the canonical step timestamp from a bar.

        Default: bar["timestamp"] (bar close time).
        """
        try:
            return float(bar["timestamp"])
        except Exception as e:
            raise KeyError("OHLCV bar must contain 'timestamp'") from e

    def _default_on_snapshot(self, snapshot: Any) -> None:
        """
        Default snapshot hook (no-op logging).
        """
        return