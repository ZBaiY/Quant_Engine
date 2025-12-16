from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json
import numpy as np
import pandas as pd

from quant_engine.data.protocol_historical import HistoricalDataHandler
from quant_engine.utils.logger import get_logger, log_debug

_logger = get_logger(__name__)


def _ensure_timestamp_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has float seconds column named 'timestamp'.
    Accepts: 'timestamp' or 'ts' or datetime64 columns.
    """
    out = df.copy()

    if "timestamp" not in out.columns:
        if "ts" in out.columns:
            out["timestamp"] = out["ts"]
        else:
            raise KeyError("Sentiment historical data must contain 'timestamp' (or 'ts') column")

    s = out["timestamp"]

    # datetime -> seconds
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, utc=True, errors="coerce")
        ns = dt.astype("int64")  # ns since epoch; NaT -> min int
        sec = ns.astype("float64") / 1_000_000_000.0
        out["timestamp"] = sec.where(dt.notna(), np.nan).astype("float64")
        return out

    # numeric ms/seconds -> seconds (heuristic: if > 1e12 treat as ms)
    v = pd.to_numeric(s, errors="coerce")
    # if values look like ms epoch, divide by 1000
    ms_mask = v > 1.0e12
    out["timestamp"] = v.astype("float64")
    out.loc[ms_mask, "timestamp"] = out.loc[ms_mask, "timestamp"] / 1000.0
    return out


def _coerce_embedding(x: Any) -> List[float] | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, list):
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8")
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip()
        # common storage: JSON string like "[0.1, 0.2, ...]"
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [float(v) for v in arr]
            except Exception:
                return None
    return None


def _coerce_meta(x: Any) -> Dict[str, Any]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        # JSON object string
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return {}
    return {}


@dataclass
class SentimentRow:
    timestamp: float
    score: float
    embedding: List[float] | None
    meta: Dict[str, Any]
    source: str | None
    model: str | None

    def to_payload(self) -> Dict[str, Any]:
        # 兼容 SentimentHandler._coerce_sentiment_snapshot(...)
        return {
            "obs_ts": float(self.timestamp),
            "score": float(self.score),
            "embedding": self.embedding,
            "meta": self.meta,
            "source": self.source,
            "model": self.model,
        }


class HistoricalSentimentHandler(HistoricalDataHandler):
    """
    Historical sentiment *signal source* for BACKTEST replay (read-only).

    Storage layout (suggested, matches your OHLCV ingestion style):
      data_root/
        sentiment/
          <kind>/                # e.g. cleaned/raw
            <symbol>/
              <interval>/
                2024.parquet
                2025.parquet

    Required columns (minimum):
      - timestamp (seconds or ms epoch or datetime64)
      - score (float)
    Optional:
      - embedding (list or JSON string)
      - meta (dict or JSON string)
      - source, model (strings)
    """

    symbol: str
    interval: str
    data_root: Path
    kind: str

    def __init__(
        self,
        *,
        symbol: str,
        **kwargs: Any
    ) -> None:
        self.symbol = symbol

        itv = kwargs.get("interval")
        assert isinstance(itv, str) and itv
        self.interval = itv
        
        dr = kwargs.get("data_root", kwargs.get("root"))
        assert dr is not None
        self.data_root = Path(dr)
        self.kind = kwargs.get("kind", "cleaned")

        self._df: pd.DataFrame | None = None
        self._loaded_range: tuple[float, float | None] | None = None

        log_debug(_logger, "HistoricalSentimentHandler initialized",
                  symbol=symbol, interval=self.interval, kind=self.kind, data_root=str(self.data_root))

    # ---------- optional helper for StrategyEngine.load_history() ----------
    def load_history(self, *, start_ts: float, end_ts: float | None = None) -> None:
        self._df = self._load_range_df(start_ts=start_ts, end_ts=end_ts)
        self._loaded_range = (float(start_ts), None if end_ts is None else float(end_ts))

    # ---------- HistoricalSignalSource contract ----------
    def last_timestamp(self) -> float | None:
        df = self._df
        if df is None or df.empty:
            return None
        t = df["timestamp"].iloc[-1]
        return None if pd.isna(t) else float(t)

    def window(self, ts: float | None = None, n: int = 1) -> Any:
        df = self._df
        if df is None:
            # window() is called before load_history in some tests -> load minimal by scanning all years is expensive
            # So we return empty to force caller to call load_history first.
            return []

        if df.empty:
            return []

        n = int(n)
        if n <= 0:
            return []

        if ts is None:
            sub = df
        else:
            sub = df[df["timestamp"] <= float(ts)]

        if sub.empty:
            return []

        tail = sub.tail(n)

        out: list[Dict[str, Any]] = []
        for _, r in tail.iterrows():
            row = SentimentRow(
                timestamp=float(r["timestamp"]),
                score=float(r.get("score", 0.0) if not pd.isna(r.get("score", np.nan)) else 0.0),
                embedding=_coerce_embedding(r.get("embedding")),
                meta=_coerce_meta(r.get("meta")),
                source=(r.get("source") if isinstance(r.get("source"), str) else None),
                model=(r.get("model") if isinstance(r.get("model"), str) else None),
            )
            out.append(row.to_payload())
        return out

    def iter_range(self, *, start_ts: float, end_ts: float | None = None) -> Iterable[Any]:
        df = self._df
        if df is None:
            # same policy: require load_history
            return iter(())

        if df.empty:
            return iter(())

        start = float(start_ts)
        if end_ts is None:
            sub = df[df["timestamp"] >= start]
        else:
            end = float(end_ts)
            sub = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

        def _gen() -> Iterable[Dict[str, Any]]:
            for _, r in sub.iterrows():
                row = SentimentRow(
                    timestamp=float(r["timestamp"]),
                    score=float(r.get("score", 0.0) if not pd.isna(r.get("score", np.nan)) else 0.0),
                    embedding=_coerce_embedding(r.get("embedding")),
                    meta=_coerce_meta(r.get("meta")),
                    source=(r.get("source") if isinstance(r.get("source"), str) else None),
                    model=(r.get("model") if isinstance(r.get("model"), str) else None),
                )
                yield row.to_payload()

        return _gen()

    # ---------- IO: local parquet only (still read-only / no crawling) ----------
    def _base_dir(self) -> Path:
        return self.data_root / "sentiment" / self.kind / self.symbol / self.interval

    def _year_path(self, year: int) -> Path:
        return self._base_dir() / f"{year}.parquet"

    def _load_range_df(self, *, start_ts: float, end_ts: float | None) -> pd.DataFrame:
        start_dt = pd.to_datetime(float(start_ts), unit="s", utc=True)
        end_dt = pd.to_datetime(float(end_ts), unit="s", utc=True) if end_ts is not None else None

        years = {int(start_dt.year)}
        if end_dt is not None:
            years |= set(range(int(start_dt.year), int(end_dt.year) + 1))

        frames: list[pd.DataFrame] = []
        for y in sorted(years):
            p = self._year_path(y)
            if not p.exists():
                continue
            try:
                frames.append(pd.read_parquet(p))
            except Exception as e:
                log_debug(_logger, "Failed reading sentiment parquet", path=str(p), error=str(e))

        if not frames:
            return pd.DataFrame(columns=["timestamp", "score", "embedding", "meta", "source", "model"])

        df = pd.concat(frames, ignore_index=True)
        df = _ensure_timestamp_seconds(df).dropna(subset=["timestamp"]).sort_values("timestamp")

        start = float(start_ts)
        if end_ts is None:
            df = df[df["timestamp"] >= start]
        else:
            end = float(end_ts)
            df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

        df = df.reset_index(drop=True)
        return df