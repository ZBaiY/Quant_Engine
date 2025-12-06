# backtest/engine.py

from typing import List, Dict, Any

from quant_engine.strategy.engine import StrategyEngine
from quant_engine.data.historical import HistoricalDataHandler
from quant_engine.data.realtime import RealTimeDataHandler


class BacktestEngine:
    """
    Unified Backtest Driver (skeleton).

    This wraps:
        - HistoricalDataHandler (Layer 1 - historical)
        - RealTimeDataHandler   (Layer 1 - streaming simulator)
        - StrategyEngine        (Layer 7 orchestrator)

    The backtest engine feeds each historical bar into the
    RealTimeDataHandler as if it were streaming live data, ensuring
    backtest and live share the same execution pipeline.
    """

    def __init__(
        self,
        strategy_engine: StrategyEngine,
        historical: HistoricalDataHandler,
        realtime: RealTimeDataHandler,
    ):
        self.strategy = strategy_engine
        self.historical = historical
        self.realtime = realtime

        # Store results for analysis / reporting
        self.results: List[Dict[str, Any]] = []

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------
    def run(self) -> List[Dict[str, Any]]:
        """
        Iterate over historical bars.
        For each bar:
            (1) push into RealTimeDataHandler
            (2) call strategy.step()
            (3) collect snapshot
        """

        for bar in self.historical.iter_bars():

            # ---- Feed new bar to the realtime handler ----
            self.realtime.on_new_tick(bar)

            # ---- Execute one pipeline step ----
            snapshot = self.strategy.step()

            # ---- Record ----
            self.results.append(snapshot)

        return self.results

    # -------------------------------------------------
    # Optional: reset method
    # -------------------------------------------------
    def reset(self):
        """Reset state (optional, for repeated backtests)."""
        self.results.clear()
        self.realtime.reset()
        self.strategy.portfolio.reset()