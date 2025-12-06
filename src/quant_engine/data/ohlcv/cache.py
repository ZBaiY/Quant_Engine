import pandas as pd
from collections import deque
from quant_engine.utils.logger import get_logger, log_debug

class DataCache:
    """
    Rolling window cache for OHLCV or other bar data.
    Keeps the last N bars, used by features and models.
    """

    def __init__(self, window: int = 1000):
        self.window = window
        self.buffer = deque(maxlen=window)
        self._logger = get_logger(__name__)
        log_debug(self._logger, "DataCache initialized", window=window)

    def update(self, bar: pd.DataFrame):
        """Add the newest bar (single-row DataFrame)."""
        log_debug(self._logger, "DataCache received bar")
        self.buffer.append(bar)
        log_debug(self._logger, "DataCache updated", size=len(self.buffer))

    def get_window(self) -> pd.DataFrame:
        """Return the current rolling window as one DataFrame."""
        log_debug(self._logger, "DataCache returning window", size=len(self.buffer))
        if not self.buffer:
            return pd.DataFrame()
        return pd.concat(list(self.buffer), ignore_index=True)

    def get_latest(self) -> pd.DataFrame:
        """Return the most recent bar (single-row DataFrame)."""
        log_debug(self._logger, "DataCache returning latest", size=len(self.buffer))
        if not self.buffer:
            return pd.DataFrame()
        return self.buffer[-1]

    def clear(self):
        """Clear all cached bars."""
        log_debug(self._logger, "DataCache cleared")
        self.buffer.clear()