from quant_engine.strategy.engine import StrategyEngine
from quant_engine.data.ohlcv.realtime import RealTimeDataHandler


class LiveEngine:
    """
    Unified Live Trading Engine (skeleton).

    Responsibilities:
        • Receive real-time bars/ticks from a live feed (WebSocket / REST poll).
        • Push each update into RealTimeDataHandler.
        • Call StrategyEngine.step() on each update.
        • Let ExecutionEngine handle routing → slippage → matching (live).
    """

    def __init__(
        self,
        strategy: StrategyEngine,
        realtime: RealTimeDataHandler,
        feed=None,
    ):
        """
        feed: object providing live bars/ticks.
               Expected to yield dict-like bars:
               {
                   "timestamp": ...,
                   "open": ...,
                   "high": ...,
                   "low": ...,
                   "close": ...,
                   "volume": ...
               }
        """
        self.strategy = strategy
        self.realtime = realtime
        self.feed = feed   # WebSocket wrapper or polling generator

    def run(self):
        """
        Main live loop.
        For each bar from the live feed:
            1) realtime.on_new_tick(bar)
            2) strategy.step()
            3) handle returned snapshot (log, store, print)
        """
        if self.feed is None:
            raise ValueError("LiveEngine requires a live feed source.")

        for bar in self.feed:
            # 1) Feed the bar into the real-time handler
            self.realtime.on_new_tick(bar)

            # 2) Execute one strategy iteration
            snapshot = self.strategy.step()

            # 3) Hook for logging / printing / storage
            self.on_snapshot(snapshot)

    def on_snapshot(self, snapshot):
        """
        Override or extend this method in the future.
        For now, skeleton simply prints minimal info.
        """
        print("[LIVE] snapshot:", snapshot.get("decision_score"), snapshot.get("target_position"))