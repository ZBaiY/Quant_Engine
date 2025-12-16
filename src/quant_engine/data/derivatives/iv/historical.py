

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import json

import numpy as np
import pandas as pd

from quant_engine.utils.logger import get_logger, log_debug, log_info
from quant_engine.data.derivatives.iv.snapshot import IVSurfaceSnapshot


_logger = get_logger(__name__)


def _to_float_scalar(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        return float(x.timestamp())
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    try:
        v = float(x)
    except Exception:
        return None
    # tolerate ms epoch
    if v > 1.0e12:
        v = v / 1000.0
    return float(v)


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
            raise KeyError("IV surface historical data must contain 'timestamp' (or 'ts')")

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


def _coerce_dict_field(x: Any) -> Dict[str, Any]:
    """Coerce JSON-ish fields to dict.

    Accepts:
      - dict
      - JSON string ("{...}")
      - None/NaN -> {}
    """
    if x is None:
        return {}
    if isinstance(x, float) and np.isnan(x):
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
    return {}


def _coerce_curve_field(x: Any) -> Dict[str, float]:
    """Coerce curve field to Dict[str,float].

    Accepts:
      - dict[str, number]
      - JSON string dict
      - None/NaN -> {}
    """
    d = _coerce_dict_field(x)
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


class HistoricalIVSurfaceHandler:
    """Historical IV surface *signal source* for BACKTEST replay.

    Contract (HistoricalSignalSource / HistoricalDataHandler style):
      - symbol: str
      - last_timestamp() -> float | None
      - window(ts: float | None, n: int) -> Any
      - iter_range(start_ts: float, end_ts: float | None) -> Iterable[Any]

    Notes:
      - Read-only. No crawling/store/fit responsibilities here.
      - Yields IVSurfaceSnapshot items.

    Storage (suggested):
      data_root/iv_surface/<kind>/<symbol>/<interval>/*.parquet
      (or provide explicit `path` to a single csv/parquet)

    Expected columns (minimum):
      - timestamp (seconds/ms/datetime)
      - atm_iv
      - skew
      - curve (dict or JSON string)
    Optional:
      - surface (dict or JSON string)
      - model
      - expiry
      - symbol (if present, ignored in favor of handler.symbol)
    """

    def __init__(
        self,
        *,
        symbol: str,
        **kwargs: Any,
    ) -> None:
        self.symbol = symbol
        path = kwargs.get("path")
        data_root = kwargs.get("data_root", kwargs.get("root"))
        kind = kwargs.get("kind", "cleaned")
        interval = kwargs.get("interval")
        refresh_interval = kwargs.get("refresh_interval")
        
        ri = interval if interval is not None else refresh_interval
        if not isinstance(ri, str) or not ri:
            raise ValueError("HistoricalIVSurfaceHandler requires non-empty 'interval' (or 'refresh_interval')")
        self.interval = ri

        self.path = (str(path) if path else "")
        self.data_root = Path(data_root) if data_root is not None else None

        if not isinstance(kind, str) or not kind:
            raise ValueError("HistoricalIVSurfaceHandler 'kind' must be a non-empty string")
        self.kind = kind

        self._df: pd.DataFrame | None = None
        self._loaded_range: tuple[float, float | None] | None = None
        self._logger = get_logger(self.__class__.__name__)

        log_debug(
            self._logger,
            "HistoricalIVSurfaceHandler initialized",
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
            raise ValueError("data_root is None; cannot resolve iv_surface storage directory")
        return self.data_root / "iv_surface" / self.kind / self.symbol / self.interval

    def load(self) -> pd.DataFrame:
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
                raise ValueError(f"Unsupported iv_surface history file format: {p.suffix}")
        else:
            d = self._base_dir()
            files = sorted(d.glob("*.parquet"))
            if not files:
                files = sorted(d.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No iv_surface files found under {d}")

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
        log_info(self._logger, "HistoricalIVSurfaceHandler loaded", symbol=self.symbol, rows=int(len(df)))
        return df

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        symbol: str,
        interval: str,
        kind: str = "cleaned",
    ) -> "HistoricalIVSurfaceHandler":
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

        Returns: list[IVSurfaceSnapshot]
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

        # If we have expiry column, snapshots are grouped by (timestamp, expiry)
        group_keys: list[str]
        if "expiry" in sub.columns:
            group_keys = ["timestamp", "expiry"]
        else:
            group_keys = ["timestamp"]

        # Determine last n snapshot groups by timestamp ordering
        ts_vals = sub["timestamp"].dropna().unique()
        if len(ts_vals) == 0:
            return []
        ts_sorted = np.sort(ts_vals.astype("float64"))
        tail_ts = ts_sorted[-n_i:]

        out: list[IVSurfaceSnapshot] = []
        for t in tail_ts:
            rows_t = sub[sub["timestamp"] == float(t)]
            if rows_t.empty:
                continue

            if group_keys == ["timestamp"]:
                row0 = rows_t.iloc[-1]
                out.append(self._row_to_snapshot(engine_ts=float(t), row=row0))
            else:
                # multiple expiries at same t
                for _, rows_e in rows_t.groupby("expiry", sort=True):
                    if rows_e.empty:
                        continue
                    row0 = rows_e.iloc[-1]
                    out.append(self._row_to_snapshot(engine_ts=float(t), row=row0))

        return out

    def iter_range(self, *, start_ts: float, end_ts: float | None = None) -> Iterable[Any]:
        """Iterate IVSurfaceSnapshot items in [start_ts, end_ts]."""
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

        def _gen() -> Iterator[IVSurfaceSnapshot]:
            if "expiry" in sub.columns:
                for (snap_ts, expiry), rows in sub.groupby(["timestamp", "expiry"], sort=True):
                    t = _to_float_scalar(snap_ts)
                    if t is None or rows.empty:
                        continue
                    yield self._row_to_snapshot(engine_ts=t, row=rows.iloc[-1])
            else:
                for snap_ts, rows in sub.groupby("timestamp", sort=True):
                    t = _to_float_scalar(snap_ts)
                    if t is None or rows.empty:
                        continue
                    yield self._row_to_snapshot(engine_ts=t, row=rows.iloc[-1])

        return _gen()

    # ------------------------------------------------------------------
    # Row coercion
    # ------------------------------------------------------------------
    def _row_to_snapshot(self, *, engine_ts: float, row: Any) -> IVSurfaceSnapshot:
        if hasattr(row, "to_dict"):
            d = row.to_dict()  # type: ignore[no-any-return]
        elif isinstance(row, dict):
            d = dict(row)
        else:
            raise TypeError("row must be a pandas Series or dict-like")

        surface_ts = _to_float_scalar(d.get("timestamp", d.get("ts")))
        if surface_ts is None:
            surface_ts = float(engine_ts)

        atm_iv = d.get("atm_iv", 0.0)
        skew = d.get("skew", 0.0)
        try:
            atm_iv_f = float(atm_iv)
        except Exception:
            atm_iv_f = 0.0
        try:
            skew_f = float(skew)
        except Exception:
            skew_f = 0.0

        curve = _coerce_curve_field(d.get("curve"))
        surface_params = _coerce_dict_field(d.get("surface"))

        expiry = d.get("expiry")
        expiry_s = expiry if isinstance(expiry, str) and expiry else None

        model = d.get("model")
        model_s = model if isinstance(model, str) and model else None

        # historical context: engine_ts == surface_ts => latency=0
        return IVSurfaceSnapshot.from_surface(
            ts=float(engine_ts),
            surface_ts=float(surface_ts),
            atm_iv=float(atm_iv_f),
            skew=float(skew_f),
            curve=curve,
            surface_params=surface_params,
            symbol=self.symbol,
            expiry=expiry_s,
            model=model_s,
        )