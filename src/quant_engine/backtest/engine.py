# backtest/engine.py

from typing import List, Dict, Any

from quant_engine.strategy.engine import StrategyEngine
from quant_engine.data.ohlcv.historical import HistoricalDataHandler
from quant_engine.data.ohlcv.realtime import RealTimeDataHandler
from quant_engine.utils.logger import get_logger
from quant_engine.runtime.log_router import attach_artifact_handlers


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

    @classmethod
    def from_historical(cls, strategy_engine: StrategyEngine, historical: HistoricalDataHandler, window: int = 1000):
        """
        Convenience constructor:
        Automatically create a RealTimeDataHandler for backtesting.
        """
        realtime = RealTimeDataHandler(window=window)
        return cls(strategy_engine, historical, realtime)

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
        run_id = "2025-12-14_backtest_001"

        logger = get_logger()   # root logger

        attach_artifact_handlers(
            logger,
            run_id=run_id,
            decisions=True,
            execution=True,
            data_repairs=True,
        )
        for bar in self.historical.iter_bars():

            # ---- Feed new bar to the realtime handler ----
            window_df = self.realtime.on_new_tick(bar)
            if window_df is None or len(window_df) == 0:
                continue  # skip until we have enough data
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