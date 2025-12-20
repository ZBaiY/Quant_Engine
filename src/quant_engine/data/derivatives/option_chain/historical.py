from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
import time

import numpy as np
import pandas as pd

from quant_engine.utils.logger import get_logger, log_debug, log_info
from quant_engine.data.derivatives.option_chain.snapshot import OptionChainSnapshot


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
            raise KeyError("Option chain historical data must contain 'timestamp' (or 'ts')")

    s = out["timestamp"]

    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, utc=True, errors="coerce")
        ns = dt.astype("int64")
        sec = ns.astype("float64") / 1_000_000_000.0
        out["timestamp"] = sec.where(dt.notna(), np.nan).astype("float64")
        return out

    v = pd.to_numeric(s, errors="coerce").astype("float64")
    ms_mask = v > 1.0e12
    v.loc[ms_mask] = v.loc[ms_mask] / 1000.0
    out["timestamp"] = v
    return out


class HistoricalOptionChainHandler:
    """Historical option-chain *signal source* for BACKTEST replay.

    Contract (HistoricalSignalSource / HistoricalDataHandler style):
      - symbol: str
      - last_timestamp() -> float | None
      - window(ts: float | None, n: int) -> Any
      - iter_range(start_ts: float, end_ts: float | None) -> Iterable[Any]

    Notes:
      - This handler is read-only. No crawling/store/clean responsibilities here.
      - It yields OptionChainSnapshot items.
      - Stored data is assumed to be a *flattened contract table* per timestamp.
        (i.e. one row per option contract, with a 'timestamp' column)
    """

    def __init__(
        self,
        *,
        symbol: str,
        **kwargs: Any
    ) -> None:
        interval = kwargs.get("interval")
        path = kwargs.get("path")
        data_root = kwargs.get("data_root", kwargs.get("root"))
        kind = kwargs.get("kind", "cleaned")
        
        self.symbol = symbol
        self.interval = interval
        self.path = (str(path) if path else "")
        self.data_root = Path(data_root) if data_root is not None else None
        self.kind = kind

        self._df: pd.DataFrame | None = None
        self._loaded_range: tuple[float, float | None] | None = None
        self._logger = get_logger(self.__class__.__name__)

        log_debug(
            self._logger,
            "HistoricalOptionChainHandler initialized",
            symbol=self.symbol,
            interval=self.interval,
            kind=self.kind,
            path=self.path,
            data_root=str(self.data_root) if self.data_root is not None else None,
        )

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
        if self.data_root is None:
            raise ValueError("data_root is None; cannot resolve option_chain storage directory")
        return self.data_root / "option_chain" / self.kind / self.symbol / self.interval

    def load(self) -> pd.DataFrame:
        """Load the entire dataset from disk. Prefer load_history(...) for slicing."""
        if self._df is not None:
            return self._df

        df: pd.DataFrame

        if self.path:
            p = Path(self.path)
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
            elif p.suffix.lower() in (".parquet", ".pq"):
                df = pd.read_parquet(p)
            else:
                raise ValueError(f"Unsupported option chain history file format: {p.suffix}")
        else:
            d = self._base_dir()
            files = sorted(d.glob("*.parquet"))
            if not files:
                # fallback to csv glob
                files = sorted(d.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No option chain files found under {d}")

            frames: list[pd.DataFrame] = []
            for fp in files:
                if fp.suffix.lower() == ".csv":
                    frames.append(pd.read_csv(fp))
                else:
                    frames.append(pd.read_parquet(fp))
            df = pd.concat(frames, ignore_index=True)

        df = _ensure_timestamp_seconds(df)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        self._df = df
        log_info(self._logger, "HistoricalOptionChainHandler loaded", symbol=self.symbol, rows=int(len(df)))
        return df

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        symbol: str,
        interval: str,
        kind: str = "cleaned",
    ) -> "HistoricalOptionChainHandler":
        obj = cls(symbol=symbol, interval=interval, path=None, data_root=None, kind=kind)
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
        """Return last n snapshots up to ts (or latest if ts is None).

        Returns: list[OptionChainSnapshot]
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

        # snapshot units are grouped by timestamp
        ts_vals = sub["timestamp"].dropna().unique()
        if len(ts_vals) == 0:
            return []

        # sort unique timestamps and take tail(n)
        ts_sorted = np.sort(ts_vals.astype("float64"))
        tail_ts = ts_sorted[-n_i:]

        out: list[OptionChainSnapshot] = []
        for snap_ts in tail_ts:
            rows = sub[sub["timestamp"] == float(snap_ts)]
            # historical context: engine_ts == chain_ts => latency=0
            out.append(
                OptionChainSnapshot.from_chain_aligned(
                    timestamp=float(snap_ts),
                    data_ts=float(snap_ts),
                    symbol=self.symbol,
                    chain=rows,
                )
            )
        return out

    def iter_range(self, *, start_ts: float, end_ts: float | None = None) -> Iterable[Any]:
        """Iterate OptionChainSnapshot items in [start_ts, end_ts]."""
        df = self._df if self._df is not None else self.load()
        if df is None or df.empty:
            return iter(())

        start = float(start_ts)
        if end_ts is None:
            sub = df[df["timestamp"] >= start]
        else:
            end = float(end_ts)
            sub = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

        if sub.empty:
            return iter(())

        # group by timestamp
        def _gen() -> Iterator[OptionChainSnapshot]:
            for snap_ts, rows in sub.groupby("timestamp", sort=True):
                t = _to_float_scalar(snap_ts)  # <- snap_ts treated as Any inside helper
                if t is None:
                    continue
                yield OptionChainSnapshot.from_chain_aligned(
                    timestamp=float(t),
                    data_ts=float(t),
                    symbol=self.symbol,
                    chain=rows,
                )

        return _gen()

def _to_float_scalar(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        return float(x.timestamp())
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return None