from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from quant_engine.utils.logger import get_logger, log_debug, log_info
from quant_engine.data.ohlcv.snapshot import OHLCVSnapshot


_logger = get_logger(__name__)


def _ensure_ts_seconds_from_close_time(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to a float-seconds `timestamp` column using close_time if present.

    Accepts:
      - close_time as datetime64 / ms epoch
      - timestamp/ts as fallback

    Invariant:
      - `timestamp` is the BAR CLOSE TIME in seconds (float)
    """
    out = df.copy()

    if "close_time" in out.columns:
        s = out["close_time"]
    elif "timestamp" in out.columns:
        s = out["timestamp"]
    elif "ts" in out.columns:
        s = out["ts"]
    else:
        raise KeyError("OHLCV historical data must contain 'close_time' or 'timestamp' (or 'ts')")

    # datetime -> seconds
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, utc=True, errors="coerce")
        ns = dt.astype("int64")
        sec = ns.astype("float64") / 1_000_000_000.0
        out["timestamp"] = sec.where(dt.notna(), np.nan).astype("float64")
        return out

    v = pd.to_numeric(s, errors="coerce").astype("float64")
    # heuristic: > 1e12 means ms epoch
    ms_mask = v > 1.0e12
    v.loc[ms_mask] = v.loc[ms_mask] / 1000.0
    out["timestamp"] = v
    return out


class HistoricalOHLCVHandler:
    """Historical OHLCV *signal source* for BACKTEST replay.

    Design:
      - Read-only (no crawling / cleaning / storing here)
      - Local parquet input written by OHLCVIngestionEngine
      - Implements HistoricalSignalSource semantics via:
          - last_timestamp()
          - window(ts,n)
          - iter_range(start_ts,end_ts)

    Storage layout (matches ingestion engine):
      data_root/klines/cleaned/<symbol>/<interval>/*.parquet
      (year-partitioned parquet files are supported via glob)
    """

    def __init__(self, symbol: str, **kwargs: Any) -> None:
        self.symbol = symbol

        interval = kwargs.get("interval")
        if not isinstance(interval, str) or not interval:
            raise ValueError("HistoricalOHLCVHandler requires non-empty 'interval' (e.g. '1m')")
        self.interval = interval

        data_root = kwargs.get("data_root", kwargs.get("root", "data"))
        self.data_root = Path(data_root)

        kind = kwargs.get("kind", "cleaned")
        if not isinstance(kind, str) or not kind:
            raise ValueError("HistoricalOHLCVHandler 'kind' must be a non-empty string")
        self.kind = kind

        self._df: pd.DataFrame | None = None
        self._loaded_range: tuple[float, float | None] | None = None
        self._logger = get_logger(self.__class__.__name__)

        log_debug(self._logger, "HistoricalOHLCVHandler initialized",
                  symbol=self.symbol, interval=self.interval, kind=self.kind, data_root=str(self.data_root))

    # ------------------------------------------------------------------
    # Optional helper for StrategyEngine.load_history() (not required by protocol)
    # ------------------------------------------------------------------
    def load_history(self, *, start_ts: float, end_ts: float | None = None) -> None:
        self._df = self._load_range_df(start_ts=float(start_ts), end_ts=None if end_ts is None else float(end_ts))
        self._loaded_range = (float(start_ts), None if end_ts is None else float(end_ts))

    # ------------------------------------------------------------------
    # IO (local)
    # ------------------------------------------------------------------
    def _base_dir(self) -> Path:
        return self.data_root / "klines" / self.kind / self.symbol / self.interval

    def load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        d = self._base_dir()
        files = sorted(d.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No OHLCV parquet found under {d}")

        df = pd.concat((pd.read_parquet(p) for p in files), ignore_index=True)

        # Normalize timestamp = close_time (seconds)
        df = _ensure_ts_seconds_from_close_time(df)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        self._df = df
        log_info(self._logger, "HistoricalOHLCVHandler loaded", symbol=self.symbol, rows=int(len(df)))
        return df

    def _load_range_df(self, *, start_ts: float, end_ts: float | None) -> pd.DataFrame:
        df = self.load()
        if df.empty:
            return df

        start = float(start_ts)
        if end_ts is None:
            sub = df[df["timestamp"] >= start]
        else:
            end = float(end_ts)
            sub = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        return sub.reset_index(drop=True)

    # ------------------------------------------------------------------
    # HistoricalSignalSource
    # ------------------------------------------------------------------
    def last_timestamp(self) -> float | None:
        df = self._df if self._df is not None else self.load()
        if df is None or df.empty:
            return None
        t = df["timestamp"].iloc[-1]
        return None if pd.isna(t) else float(t)

    def window(self, ts: float | None = None, n: int = 1) -> Any:
        """Return last n bars up to ts (or latest if ts is None).

        Returns: list[dict] where each dict is a bar with keys compatible with OHLCVSnapshot.from_bar.
        """
        df = self._df if self._df is not None else self.load()
        if df is None or df.empty:
            return []

        n_i = int(n)
        if n_i <= 0:
            return []

        if ts is None:
            sub = df
        else:
            sub = df[df["timestamp"] <= float(ts)]

        if sub.empty:
            return []

        tail = sub.tail(n_i)
        # keep only canonical fields; tolerate extra columns
        cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in tail.columns]
        recs = tail[cols].to_dict(orient="records")
        return recs

    def iter_range(self, *, start_ts: float, end_ts: float | None = None) -> Iterable[Any]:
        """Iterate OHLCVSnapshot items in [start_ts, end_ts]."""
        df = self._df if self._df is not None else self.load()
        if df is None or df.empty:
            return iter(())

        start = float(start_ts)
        if end_ts is None:
            sub = df[df["timestamp"] >= start]
        else:
            end = float(end_ts)
            sub = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

        def _gen() -> Iterator[OHLCVSnapshot]:
            for _, row in sub.iterrows():
                bar = row.to_dict()
                bar_ts = float(bar.get("timestamp", 0.0) or 0.0)
                # historical context: engine ts == bar ts => latency=0
                yield OHLCVSnapshot.from_bar(ts=bar_ts, bar=bar)

        return _gen()


# Invariant:
# HistoricalOHLCVHandler.iter_range/window() must return bars whose timestamp is BAR CLOSE TIME.
