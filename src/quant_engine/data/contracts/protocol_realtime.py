from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

# Contract layer must be lightweight.
# If callers want pandas timestamps, keep it as type-only to avoid runtime dependency.
if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    TimestampLike = float | pd.Timestamp
else:
    TimestampLike = float

from quant_engine.data.contracts.protocol_historical import HistoricalSignalSource


@runtime_checkable
class DataHandlerProto(Protocol):
    """Runtime-facing handler contract consumed by Engine/FeatureExtractor.

    Notes
    -----
    - Engine/Driver own lifecycle semantics; handlers just provide the methods.
    - Backtest seeding is done via `RealTimeDataHandler.from_historical(...)` + replay ticks.
    - Some handlers may implement extra methods (e.g., load_history) but Engine should not require them.
    """

    # -------- lifecycle --------

    def bootstrap(self, *, end_ts: float, lookback: Any | None = None) -> None:
        """REALTIME/MOCK: preload recent data into cache."""
        ...

    def warmup_to(self, ts: float) -> None:
        """Align internal cursor/cache to ts. Must be idempotent."""
        ...

    # -------- runtime --------

    def last_timestamp(self) -> float | None:
        """Return latest available timestamp; None if empty."""
        ...

    def get_snapshot(self) -> Any:
        """Return latest snapshot (dict/dataclass/TypedDict). Must be serializable."""
        ...

    def window(self, ts: float | None = None, n: int = 1) -> Any:
        """Return last n snapshots up to ts (or latest if ts is None)."""
        ...


@runtime_checkable
class RealTimeDataHandler(DataHandlerProto, Protocol):
    """Runtime OHLCV handler contract.

    Design intent:
      - Runtime handler is what Engine consumes in ALL modes.
      - In BACKTEST, runtime handler is seeded from a HistoricalSignalSource and driven by replay ticks.
    """

    @classmethod
    def from_historical(
        cls,
        historical_handler: HistoricalSignalSource,
        *,
        start_ts: TimestampLike | None = None,
        window: int = 1000,
        **kwargs: Any,
    ) -> "RealTimeDataHandler":
        """Seed a runtime handler from historical data (anti-lookahead safe)."""
        ...

    def on_new_tick(self, bar: Any) -> None:
        """Push one new tick/bar into the runtime cache (live or replay)."""
        ...