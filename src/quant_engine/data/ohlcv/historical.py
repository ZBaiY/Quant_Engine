from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd






@dataclass
class HistoricalOHLCVHandler:
    """
    Runtime-compatible historical OHLCV handler.

    This handler is a deterministic, read-only data source used to:
    - seed realtime OHLCV caches before backtests
    - provide timestamp-aligned snapshots during backtests

    It MUST NOT perform streaming, mutation, or runtime data access.
    """
    data_root: Path
    symbol: str
    interval: str
    window_size: int = 500

    _cache: Optional[pd.DataFrame] = None

    def _cleaned_dir(self) -> Path:
        return self.data_root / "klines" / "cleaned" / self.symbol / self.interval

    def _load(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache

        files = sorted(self._cleaned_dir().glob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No cleaned OHLCV parquet for {self.symbol} {self.interval}"
            )

        df = pd.concat(pd.read_parquet(p) for p in files)

        # Enforce canonical index semantics (v4 invariant)
        if "open_time" in df.columns:
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            df = df.set_index("open_time")

        df = df.sort_index()

        self._cache = df
        return df

    def window_df(self, n: int) -> pd.DataFrame:
        """
        Return the last n bars without timestamp alignment.

        Used for cache warm-up when no explicit backtest start
        timestamp is provided.
        """
        df = self._load()
        return df.tail(n)

    def window_before_ts(self, ts: pd.Timestamp, n: int) -> pd.DataFrame:
        """
        Return the last n bars with open_time <= ts.

        This method is used ONLY during backtest initialization
        to seed realtime caches in an anti-lookahead-safe way.
        """
        df = self._load()
        ts = pd.Timestamp(ts, tz="UTC") if not isinstance(ts, pd.Timestamp) else ts

        window = df.loc[:ts].tail(n)
        if window.empty:
            raise ValueError(f"No OHLCV data available before {ts}")

        return window

    def snapshot(self, ts: pd.Timestamp) -> dict[str, object]:
        df = self._load()
        window = df.loc[:ts].tail(self.window_size)

        if window.empty:
            raise ValueError(f"No OHLCV data available before {ts}")

        last = window.iloc[-1]

        return {
            "timestamp": ts,
            "open": float(last["open"]),
            "high": float(last["high"]),
            "low": float(last["low"]),
            "close": float(last["close"]),
            "volume": float(last["volume"]),
            "window": window,
        }


# Invariant:
# HistoricalOHLCVHandler.snapshot() MUST match the realtime OHLCV snapshot schema.
