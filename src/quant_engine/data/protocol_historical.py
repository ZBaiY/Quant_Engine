from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable, TYPE_CHECKING

# Historical protocols are split from realtime/runtime protocols by design.
# Runtime contracts must stay lightweight (no heavy deps imported at runtime).
if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    TimestampLike = float | pd.Timestamp
else:
    TimestampLike = float


@runtime_checkable
class HistoricalSignalSource(Protocol):
    """Historical *signal source* for BACKTEST replay.

    This is the runtime-facing contract: Engine/Driver can replay ticks/bars from it.
    It must be *read-only* (no download/store/clean responsibilities here).
    """

    symbol: str

    def last_timestamp(self) -> float | None:
        """Latest timestamp available in the source; None if empty."""
        ...

    def window(self, ts: float | None = None, n: int = 1) -> Any:
        """Return last n items up to ts (or latest if ts is None)."""
        ...

    def iter_range(self, *, start_ts: float, end_ts: float | None = None) -> Iterable[Any]:
        """Iterate items in [start_ts, end_ts] (end_ts inclusive/exclusive is implementation-defined).

        Driver should treat this as the canonical backtest feed.
        """
        ...


@runtime_checkable
class HistoricalIngestionEngine(Protocol):
    """OFFLINE-only contract: crawl/store/clean historical datasets.

    This is NOT used by StrategyEngine runtime step().
    Keep this separate so runtime can stay deterministic and light.
    """

    def fetch_range(self, *, symbol: str, interval: str, start_ts: float, end_ts: float) -> Any:
        ...

    def clean(self, raw: Any, **kwargs: Any) -> Any:
        ...

    def store(self, *, kind: str, symbol: str, interval: str, data: Any, **kwargs: Any) -> None:
        ...

    def load_range(
        self,
        *,
        kind: str,
        symbol: str,
        interval: str,
        start_ts: float,
        end_ts: float | None = None,
        **kwargs: Any,
    ) -> Any:
        ...


@runtime_checkable
class HistoricalDataHandler(HistoricalSignalSource, Protocol):
    """A historical handler primarily as a runtime source (replay).

    Concrete historical handlers may optionally implement HistoricalIngestionEngine.
    """

    pass
