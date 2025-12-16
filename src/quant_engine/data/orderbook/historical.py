from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from quant_engine.data.orderbook.snapshot import OrderbookSnapshot
from quant_engine.utils.logger import get_logger, log_debug, log_info


_logger = get_logger(__name__)


def _ensure_timestamp_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a float-seconds `timestamp` column.

    Accepts:
      - `timestamp` as float seconds
      - `timestamp` as ms epoch
      - `timestamp` as datetime64
      - `ts` alias
    """
    out = df.copy()

    if "timestamp" not in out.columns:
        if "ts" in out.columns:
            out["timestamp"] = out["ts"]
        else:
            raise KeyError("Orderbook historical data must contain 'timestamp' (or 'ts') column")

    s = out["timestamp"]

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


def _row_to_tick(symbol: str, row: Any) -> Dict[str, Any]:
    """Coerce a DataFrame row (Series) into the tick dict schema expected by OrderbookSnapshot."""
    # tolerate Series / Mapping
    if hasattr(row, "to_dict"):
        d = row.to_dict()  # type: ignore[no-any-return]
    elif isinstance(row, dict):
        d = dict(row)
    else:
        raise TypeError("row must be a pandas Series or dict-like")

    # common aliases
    ts = d.get("timestamp", d.get("ts"))
    best_bid = d.get("best_bid", d.get("bid"))
    best_bid_size = d.get("best_bid_size", d.get("bid_size", d.get("bidQty")))
    best_ask = d.get("best_ask", d.get("ask"))
    best_ask_size = d.get("best_ask_size", d.get("ask_size", d.get("askQty")))

    tick: Dict[str, Any] = {
        "timestamp": ts,
        "best_bid": best_bid,
        "best_bid_size": best_bid_size,
        "best_ask": best_ask,
        "best_ask_size": best_ask_size,
        "bids": d.get("bids", []),
        "asks": d.get("asks", []),
    }

    # Keep symbol external (OrderbookSnapshot stores it explicitly)
    tick["symbol"] = symbol
    return tick


class HistoricalOrderbookHandler:
    """Historical orderbook *signal source* for BACKTEST replay.

    Contract (HistoricalSignalSource / HistoricalDataHandler):
      - symbol: str
      - last_timestamp() -> float | None
      - window(ts: float | None, n: int) -> Any
      - iter_range(start_ts: float, end_ts: float | None) -> Iterable[Any]

    Design:
      - Read-only (no crawling / ingestion here)
      - Uses local CSV/Parquet as storage input
      - Returns OrderbookSnapshot items
    """

    def __init__(self, symbol: str, **kwargs: Any) -> None:
        path = kwargs.get("path")
        window = kwargs.get("window", 200)
        self.path = path
        self.symbol = symbol
        # `window` is kept for backward-compat in constructor, but this class is not a cache.
        self._default_window = int(window)

        self._df: pd.DataFrame | None = None
        self._loaded_range: tuple[float, float | None] | None = None
        self._logger = get_logger(self.__class__.__name__)

        log_debug(self._logger, "HistoricalOrderbookHandler initialized", symbol=symbol, path=path, window=window)

    # ------------------------------------------------------------------
    # Optional helper for StrategyEngine.load_history() (not required by protocol)
    # ------------------------------------------------------------------
    def load_history(self, *, start_ts: float, end_ts: float | None = None) -> None:
        self._df = self._load_range_df(start_ts=float(start_ts), end_ts=None if end_ts is None else float(end_ts))
        self._loaded_range = (float(start_ts), None if end_ts is None else float(end_ts))

    # ------------------------------------------------------------------
    # IO (local)
    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Load the entire dataset from disk. Prefer load_history(...) for slicing."""
        if self._df is not None:
            return self._df

        if not self.path:
            # allow external injection via load_history/from_dataframe pattern
            self._df = pd.DataFrame(columns=[
                "timestamp",
                "best_bid",
                "best_bid_size",
                "best_ask",
                "best_ask_size",
                "bids",
                "asks",
            ])
            return self._df

        p = Path(self.path)
        log_debug(self._logger, "HistoricalOrderbookHandler loading", path=str(p))

        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        elif p.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(p)
        else:
            raise ValueError(f"Unsupported orderbook history file format: {p.suffix}")

        df = _ensure_timestamp_seconds(df).dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        self._df = df

        log_info(self._logger, "HistoricalOrderbookHandler loaded", rows=int(len(df)), symbol=self.symbol)
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, symbol: str, window: int = 200) -> "HistoricalOrderbookHandler":
        obj = cls(path="", symbol=symbol, window=window)
        obj._df = _ensure_timestamp_seconds(df).dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return obj

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
        """Return last n items up to ts (or latest if ts is None)."""
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
        out: list[OrderbookSnapshot] = []
        for _, row in tail.iterrows():
            tick = _row_to_tick(self.symbol, row)
            tick_ts = float(tick.get("timestamp", 0.0) or 0.0)
            # Use ts=tick_ts so latency=0 in historical context
            out.append(OrderbookSnapshot.from_tick(ts=tick_ts, tick=tick, symbol=self.symbol))
        return out

    def iter_range(self, *, start_ts: float, end_ts: float | None = None) -> Iterable[Any]:
        """Iterate items in [start_ts, end_ts]."""
        df = self._df if self._df is not None else self.load()
        if df is None or df.empty:
            return iter(())

        start = float(start_ts)
        if end_ts is None:
            sub = df[df["timestamp"] >= start]
        else:
            end = float(end_ts)
            sub = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

        def _gen() -> Iterator[OrderbookSnapshot]:
            for _, row in sub.iterrows():
                tick = _row_to_tick(self.symbol, row)
                tick_ts = float(tick.get("timestamp", 0.0) or 0.0)
                yield OrderbookSnapshot.from_tick(ts=tick_ts, tick=tick, symbol=self.symbol)

        return _gen()

    # ------------------------------------------------------------------
    # Legacy helpers (kept thin)
    # ------------------------------------------------------------------
    def latest_snapshot(self) -> OrderbookSnapshot | None:
        df = self._df if self._df is not None else self.load()
        if df is None or df.empty:
            return None
        row = df.iloc[-1]
        tick = _row_to_tick(self.symbol, row)
        tick_ts = float(tick.get("timestamp", 0.0) or 0.0)
        return OrderbookSnapshot.from_tick(ts=tick_ts, tick=tick, symbol=self.symbol)

    def stream(self) -> Iterable[Any]:
        """Deprecated: historical is a signal source; prefer iter_range(...) in Driver."""
        raise NotImplementedError("HistoricalOrderbookHandler.stream() is deprecated; use iter_range(start_ts,end_ts)")