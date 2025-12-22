# backtest/engine.py

import time

from quant_engine.strategy.engine import StrategyEngine

from quant_engine.utils.logger import get_logger
from quant_engine.runtime.log_router import attach_artifact_handlers


# NOTE:
# BacktestEngine assumes EngineSpec exposes a deterministic
# clock advancement method, e.g.:
#     spec.advance(ts) -> next_ts
# This encodes the strategy observation interval (not data frequency).


class BacktestEngine:
    """
    Deterministic Backtest Driver.
    """

    def __init__(
        self,
        *,
        engine: StrategyEngine,
        start_ts: float,
        end_ts: float,
        run_id: str | None = None,
    ):
        # BacktestEngine is a driver; it assumes the engine is configured for backtest semantics.

        self.engine = engine
        # Time-range is a driver concern; it must not live in EngineSpec.
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.run_id = run_id or f"backtest_{int(time.time())}"

        self._logger = get_logger(__name__)
        attach_artifact_handlers(self._logger, run_id=self.run_id)

    # -------------------------------------------------
    # Lifecycle
    # -------------------------------------------------
    def run(self) -> None:
        """
        Execute the backtest deterministically.
        """

        self._logger.info(
            "Backtest started",
            extra={"context": {
                "start_ts": self.start_ts,
                "end_ts": self.end_ts,
            }},
        )

        self.engine.preload_data()

        # Feature warmup (state initialization only)
        self.engine.warmup_features()

        # -------------------------------------------------
        # Driver-owned deterministic clock
        # -------------------------------------------------
        ts = self.start_ts
        prev_ts: float | None = None
        steps = 0

        while ts <= self.end_ts:
            snapshot = self.engine.step(ts=ts)

            # Engine must echo back the same timestamp
            snap_ts = snapshot.get("timestamp")
            if snap_ts is None:
                raise RuntimeError(
                    "StrategyEngine.step(ts) must return a snapshot "
                    "with the same 'timestamp' in BACKTEST mode"
                )

            try:
                snap_ts_f = float(snap_ts)
            except Exception as e:
                raise RuntimeError(
                    f"Invalid snapshot timestamp: {snap_ts!r}"
                ) from e

            if snap_ts_f != ts:
                raise RuntimeError(
                    f"Engine timestamp mismatch: driver ts={ts}, "
                    f"engine returned ts={snap_ts_f}"
                )

            # Advance deterministic clock using primary interval
            ts = self.engine.spec.advance(ts)
            steps += 1

        self._logger.info("Backtest completed")
