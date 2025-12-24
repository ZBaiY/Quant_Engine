from __future__ import annotations
from typing import Iterable

from quant_engine.runtime.driver import BaseDriver
from quant_engine.runtime.modes import EngineSpec
from quant_engine.strategy.engine import StrategyEngine


class BacktestDriver(BaseDriver):
    """
    Deterministic backtest driver (v4).

    Responsibilities:
      - Define a finite, deterministic engine-time timeline.
      - Advance engine time deterministically for backtests.
      - Delegate orchestration to BaseDriver.run().
    """

    def __init__(
        self,
        *,
        engine: StrategyEngine,
        spec: EngineSpec,
        start_ts: float,
        end_ts: float,
    ):
        super().__init__(engine=engine, spec=spec)
        self.start_ts = float(start_ts)
        self.end_ts = float(end_ts)

    # -------------------------------------------------
    # Time progression
    # -------------------------------------------------

    def iter_timestamps(self) -> Iterable[float]:
        """
        Yield deterministic engine-time timestamps.
        """
        timestamp = self.start_ts
        while timestamp <= self.end_ts:
            yield timestamp
            timestamp = self.spec.advance(timestamp)
