from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional
from quant_engine.runtime.lifecycle import LifecycleGuard, RuntimePhase
from quant_engine.runtime.modes import EngineSpec
from quant_engine.strategy.engine import StrategyEngine
from quant_engine.runtime.snapshot import EngineSnapshot


class BaseDriver(ABC):
    """
    Base class for all runtime drivers.

    Responsibilities:
      - Own runtime lifecycle ordering.
      - Own time progression only (no ingestion).
      - Never own strategy logic.
    """

    def __init__(self, *, engine: StrategyEngine, spec: EngineSpec):
        self.engine = engine
        self.spec = spec
        self.guard = LifecycleGuard()

        # Runtime-owned snapshots (optional)
        self._snapshots: List[EngineSnapshot] = []

    @property
    def snapshots(self) -> List[EngineSnapshot]:
        """
        Collected EngineSnapshot objects produced during runtime.

        Semantics:
          - Owned by Driver.
          - Empty if snapshots are not collected.
          - Intended for backtest / debug / artifact layers.
        """
        return self._snapshots

    # -------------------------------------------------
    # Hooks for subclasses
    # -------------------------------------------------

    @abstractmethod
    def iter_timestamps(self) -> Iterable[float]:
        """
        Yield engine-time timestamps in strictly increasing order.
        """
        raise NotImplementedError

    # -------------------------------------------------
    # Canonical runtime loop
    # -------------------------------------------------

    def run(self) -> None:
        """
        Execute the canonical v4 runtime loop.

        PRELOAD → WARMUP → STEP* → FINISH
        """

        # -------- preload --------
        self.guard.enter(RuntimePhase.PRELOAD)
        self.engine.preload_data(anchor_ts=self.spec.timestamp)

        # -------- warmup --------
        self.guard.enter(RuntimePhase.WARMUP)
        self.engine.warmup_features(anchor_ts=self.spec.timestamp)

        # -------- main loop --------
        for ts in self.iter_timestamps():
            # step (time alignment only)
            self.guard.enter(RuntimePhase.STEP)
            self.engine.align_to(ts)

            result = self.engine.step(ts=ts)

            # Optionally collect runtime snapshot
            if result is not None and isinstance(result, dict):
                snapshot = EngineSnapshot(
                    timestamp=ts,
                    mode=self.spec.mode,
                    features=result.get("features", {}),
                    model_outputs=result.get("model_outputs", {}),
                    decision_score=result.get("decision_score"),
                    target_position=result.get("target_position"),
                    fills=result.get("fills", []),
                    market_data=result.get("market_data"),
                    portfolio=result.get("portfolio", self.engine.portfolio.state()),
                )
                self._snapshots.append(snapshot)
            elif result is not None and isinstance(result, EngineSnapshot):
                self._snapshots.append(result) 
        # -------- finish --------
        self.guard.enter(RuntimePhase.FINISH)
