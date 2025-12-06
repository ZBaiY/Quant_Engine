import pandas as pd
from .cache import DataCache
import time
from quant_engine.utils.logger import get_logger, log_debug, log_info

class RealTimeDataHandler:
    """
    Real-time handler that receives bars one-by-one (mock or exchange adapter).
    """

    def __init__(self, window: int = 1000):
        self.cache = DataCache(window=window)
        self._logger = get_logger(__name__)
        log_debug(self._logger, "RealTimeDataHandler initialized", window=window)

    def on_new_tick(self, bar: pd.DataFrame):
        """
        Called when a new bar arrives from exchange or websocket.
        """
        log_debug(self._logger, "RealTimeDataHandler received tick")
        self.cache.update(bar)
        log_debug(self._logger, "RealTimeDataHandler updated cache")
        return self.cache.get_window()

    def window_df(self):
        return self.cache.get_window()

    def latest_tick(self):
        log_debug(self._logger, "RealTimeDataHandler retrieving latest tick")
        try:
            return self.cache.get_latest()
        except AttributeError:
            return self.cache.get_window().iloc[-1:].copy()

    def reset(self):
        log_info(self._logger, "RealTimeDataHandler reset requested")
        try:
            self.cache.clear()
        except AttributeError:
            pass

    def run_mock(self, df: pd.DataFrame, delay=1.0):
        """
        A mock real-time stream for testing without exchange.
        """
        log_info(self._logger, "RealTimeDataHandler starting mock stream", rows=len(df), delay=delay)
        for _, row in df.iterrows():
            bar = row.to_frame().T
            log_debug(self._logger, "RealTimeDataHandler mock tick")
            window = self.on_new_tick(bar)
            yield bar, window
            time.sleep(delay)