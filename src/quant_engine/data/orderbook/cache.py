import pandas as pd
from collections import deque
from quant_engine.utils.logger import get_logger, log_debug

class OrderbookCache:
    """
    Rolling window cache for L1/L2 orderbook snapshots.

    Snapshot structure expected:
    OrderbookSnapshot objects.

    Stored as a deque of OrderbookSnapshot.
    """

    def __init__(self, window: int = 200):
        self.window = window
        self.buffer = deque(maxlen=window)
        self._logger = get_logger(__name__)
        log_debug(self._logger, "OrderbookCache initialized", window=window)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, snapshot):
        """Append new L1/L2 snapshot (OrderbookSnapshot)."""
        log_debug(self._logger, "OrderbookCache received snapshot")
        self.buffer.append(snapshot)
        log_debug(self._logger, "OrderbookCache updated", size=len(self.buffer))

    # ------------------------------------------------------------------
    # Latest snapshot
    # ------------------------------------------------------------------
    def get_snapshot(self):
        """Return latest snapshot object."""
        if not self.buffer:
            return None
        return self.buffer[-1]

    # ------------------------------------------------------------------
    # Entire window
    # ------------------------------------------------------------------
    def get_window(self):
        """Return entire window as list of OrderbookSnapshot objects."""
        return list(self.buffer)

    # ------------------------------------------------------------------
    # Convert window to DataFrame
    # ------------------------------------------------------------------
    def window_df(self):
        rows = []
        for snap in self.buffer:
            rows.append({
                "timestamp": snap.timestamp,
                "best_bid": snap.best_bid,
                "best_bid_size": snap.best_bid_size,
                "best_ask": snap.best_ask,
                "best_ask_size": snap.best_ask_size,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # last n snapshots
    # ------------------------------------------------------------------
    def get_last_n(self, n: int):
        if not self.buffer:
            return []
        return list(self.buffer)[-n:]

    # ------------------------------------------------------------------
    # Timestamp helpers
    # ------------------------------------------------------------------
    def last_timestamp(self):
        snap = self.get_snapshot()
        if snap is None:
            return None
        return snap.timestamp

    def has_new_snapshot(self, prev_ts):
        ts = self.last_timestamp()
        return ts is not None and ts != prev_ts

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def clear(self):
        log_debug(self._logger, "OrderbookCache cleared")
        self.buffer.clear()

    def latest(self):
        return self.buffer[-1] if self.buffer else None