import time
from quant_engine.data.orderbook.cache import OrderbookCache
from quant_engine.data.orderbook.snapshot import OrderbookSnapshot
from quant_engine.utils.logger import get_logger, log_debug, log_info
from quant_engine.data.orderbook.historical import HistoricalOrderbookHandler


class RealTimeOrderbookHandler:
    """
    Real-time L1/L2 orderbook handler.
    Receives snapshots from:
        - websocket
        - mock streams (for backtest simulation)
        - matching engine feedback (live trading)

    Mirrors the structure of RealTimeDataHandler (OHLCV),
    but operates on OrderbookSnapshot objects instead of DataFrames.
    """

    def __init__(self, symbol: str, window: int = 200):
        self.symbol = symbol
        self.cache = OrderbookCache(window=window)
        self._logger = get_logger(__name__)
        log_debug(self._logger, "RealTimeOrderbookHandler initialized", symbol=symbol, window=window)
    
    @classmethod
    def from_historical(cls, historical_handler: HistoricalOrderbookHandler):
        """
        Build a RealTimeOrderbookHandler from a HistoricalOrderbookHandler.
        Preloads existing historical snapshots into realtime cache.
        """
        rt = cls(
            symbol=historical_handler.symbol,
            window=historical_handler.cache.window,
        )
        for snapshot in historical_handler.cache.get_window():
            rt.cache.update(snapshot)
        return rt
    
    # ------------------------------------------------------------------
    def on_new_snapshot(self, snapshot: OrderbookSnapshot):
        """
        Push a new orderbook snapshot (from exchange or mock source).
        """
        log_debug(self._logger, "RealTimeOrderbookHandler received snapshot")
        self.cache.update(snapshot)
        log_debug(self._logger, "RealTimeOrderbookHandler cache updated")
        return self.cache.get_window()

    # ------------------------------------------------------------------
    def latest_snapshot(self):
        """Return most recent snapshot object."""
        log_debug(self._logger, "RealTimeOrderbookHandler latest_snapshot() called")
        return self.cache.latest()

    # ------------------------------------------------------------------
    def last_timestamp(self):
        """Return timestamp of the most recent snapshot."""
        snap = self.cache.latest()
        if snap is None:
            return None
        return snap.timestamp

    # ------------------------------------------------------------------
    def window(self, n: int | None = None):
        """Return rolling orderbook window (list of snapshots)."""
        window = self.cache.get_window()
        if n is not None:
            return window[-n:]
        return window

    # ------------------------------------------------------------------
    def reset(self):
        log_info(self._logger, "RealTimeOrderbookHandler reset requested")
        self.cache.clear()

    # ------------------------------------------------------------------
    def run_mock(self, df, delay=1.0):
        """
        Simulated stream for testing (df must contain L1/L2 columns).
        Each row becomes an OrderbookSnapshot.
        """
        log_info(self._logger, "RealTimeOrderbookHandler starting mock stream", rows=len(df), delay=delay)

        for _, row in df.iterrows():
            snapshot = OrderbookSnapshot.from_dataframe(row.to_frame().T, symbol=self.symbol)
            window = self.on_new_snapshot(snapshot)
            yield snapshot, window
            time.sleep(delay)
