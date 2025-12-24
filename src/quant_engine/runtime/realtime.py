from __future__ import annotations

import time
from quant_engine.runtime.driver import BaseDriver
from quant_engine.runtime.modes import EngineSpec
from quant_engine.strategy.engine import StrategyEngine
from typing import Iterable


class RealtimeDriver(BaseDriver):
    """
    Realtime trading driver (v4).

    Semantics:
      - Engine-time advances according to EngineSpec.advance().
      - Runtime loop is open-ended.
      - Ingestion is external to runtime (apps / wiring layer).
    """

    def __init__(
        self,
        *,
        engine: StrategyEngine,
        spec: EngineSpec,
    ):
        super().__init__(engine=engine, spec=spec)

    # -------------------------------------------------
    # Time progression
    # -------------------------------------------------

    def iter_timestamps(self) -> Iterable[float]:
        """
        Yield engine-time timestamps indefinitely.

        Uses wall-clock time as the lower bound and advances
        deterministically using EngineSpec.advance().
        """
        timestamp = self.spec.timestamp or time.time()
        while True:
            yield timestamp
            timestamp = self.spec.advance(timestamp)