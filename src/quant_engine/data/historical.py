import pandas as pd
from .cache import DataCache
from quant_engine.utils.logger import get_logger, log_debug, log_info

class HistoricalDataHandler:
    """
    Load historical OHLCV data, clean it, and feed bars into cache.
    """

    def __init__(self, path: str, window: int = 1000):
        self.path = path
        self.cache = DataCache(window=window)
        self.data = None
        self._logger = get_logger(__name__)

    def load(self):
        """Load CSV or Parquet historical data."""
        log_debug(self._logger, "HistoricalDataHandler loading file", path=self.path)
        if self.path.endswith(".csv"):
            self.data = pd.read_csv(self.path)
        else:
            raise ValueError("Unsupported file format")

        # ensure timestamps sorted
        self.data = self.data.sort_values("timestamp")
        log_info(self._logger, "HistoricalDataHandler loaded data", rows=len(self.data))

        return self.data

    def iter_bars(self):
        """
        Iterate through historical data one bar at a time.
        Yields each bar as a single-row DataFrame.
        """
        log_debug(self._logger, "HistoricalDataHandler iterating bars")
        if self.data is None:
            self.load()
        assert self.data is not None
        for _, row in self.data.iterrows():
            yield row.to_frame().T

    def stream(self):
        """
        Generator: yield one bar at a time,
        update cache, and return rolling window.
        """
        log_debug(self._logger, "HistoricalDataHandler streaming bars")
        if self.data is None:
            self.load()
            
        assert self.data is not None
        for _, row in self.data.iterrows():
            bar = row.to_frame().T
            self.cache.update(bar)
            log_debug(self._logger, "HistoricalDataHandler updated cache", latest_timestamp=row["timestamp"])
            yield bar, self.cache.get_window()