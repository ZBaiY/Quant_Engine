from __future__ import annotations
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from quant_engine.utils.logger import get_logger, log_debug

_logger = get_logger(__name__)


class SentimentLoader:
    """
    v4 Sentiment Loader / Handler
    -----------------------------
    A unified component that:

      • loads raw sentiment data (CSV / DataFrame / dict-list)
      • standardizes schema
      • stores a timestamp-sorted buffer of sentiment entries
      • exposes a SentimentHandler-compatible API:

            - symbol (property)
            - latest_score()
            - get_snapshot(ts)
            - window(ts, n)
            - ready()
            - last_timestamp()
            - flush_cache()

    Notes
    -----
    - Each entry is stored internally as:

          {
              "timestamp": float,
              "value": float | dict[str, Any],
              "source": str | None,
              "meta": dict[str, Any],
          }

    - `value` is what FeatureChannels see via latest_score/get_snapshot/window.
    """

    REQUIRED_COLUMNS = ["timestamp", "value"]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        symbol: str,
        values: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._symbol = symbol
        self._values: List[Dict[str, Any]] = []

        if values:
            self._values = self._normalize_list(values)
            self._values.sort(key=lambda r: float(r["timestamp"]))

        log_debug(
            _logger,
            "SentimentLoader initialized",
            symbol=symbol,
            count=len(self._values),
        )

    # ------------------------------------------------------------------
    # Pure loading helpers (similar to OrderbookLoader)
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        symbol: str,
        path: str,
        ts_col: str = "timestamp",
        value_col: str = "value",
        source_col: Optional[str] = "source",
        meta_cols: Optional[List[str]] = None,
    ) -> "SentimentLoader":
        """
        Load sentiment series from CSV and construct a SentimentLoader.

        Expected minimal columns:
            - ts_col     (default: "timestamp")
            - value_col  (default: "value")

        Optionally:
            - source_col (e.g. "finbert", "vader", "news")
            - meta_cols  (extra columns stored in `meta` dict)
        """
        log_debug(_logger, "Loading sentiment CSV", path=path)
        df = pd.read_csv(path)
        values = cls._standardize_df(df, ts_col, value_col, source_col, meta_cols)
        return cls(symbol=symbol, values=values)

    @classmethod
    def from_dataframe(
        cls,
        symbol: str,
        df: pd.DataFrame,
        ts_col: str = "timestamp",
        value_col: str = "value",
        source_col: Optional[str] = "source",
        meta_cols: Optional[List[str]] = None,
    ) -> "SentimentLoader":
        log_debug(_logger, "Loading sentiment DataFrame", rows=len(df))
        values = cls._standardize_df(df, ts_col, value_col, source_col, meta_cols)
        return cls(symbol=symbol, values=values)

    @classmethod
    def from_dict_list(
        cls,
        symbol: str,
        snapshots: List[Dict[str, Any]],
    ) -> "SentimentLoader":
        """
        Raw API-like input:

            [
              {"timestamp": ..., "value": ..., "source": "...", ...},
              ...
            ]
        """
        log_debug(_logger, "Loading sentiment dict-list", count=len(snapshots))
        df = pd.DataFrame(snapshots)
        # Try to infer columns; fall back to defaults.
        ts_col = "timestamp"
        value_col = "value"
        source_col = "source" if "source" in df.columns else None
        meta_cols = [
            c for c in df.columns
            if c not in (ts_col, value_col) and c != (source_col or "")
        ] or None
        values = cls._standardize_df(df, ts_col, value_col, source_col, meta_cols)
        return cls(symbol=symbol, values=values)

    # ------------------------------------------------------------------
    # Internal standardization
    # ------------------------------------------------------------------
    @classmethod
    def _standardize_df(
        cls,
        df: pd.DataFrame,
        ts_col: str,
        value_col: str,
        source_col: Optional[str],
        meta_cols: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        missing = [c for c in [ts_col, value_col] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing sentiment columns: {missing}")

        df = df.copy()

        df[ts_col] = df[ts_col].astype(float)

        def normalize_value(x: Any) -> Union[float, Dict[str, Any]]:
            # simplest: if it's numeric, treat as float sentiment score
            if isinstance(x, (int, float)):
                return float(x)
            # if it's a dict-like, keep as-is (richer NLP output)
            if isinstance(x, dict):
                return x
            # if it's a string that looks like dict / JSON, very light parsing
            if isinstance(x, str):
                try:
                    parsed = eval(x)  # assume pre-sanitized internal data
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
                # fallback: store as raw text in dict
                return {"text": x}
            # unknown type → store inside dict
            return {"value": x}

        df["__value_norm__"] = df[value_col].apply(normalize_value)

        meta_cols = meta_cols or []
        for col in meta_cols:
            if col not in df.columns:
                meta_cols.remove(col)

        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            meta: Dict[str, Any] = {}
            for col in meta_cols:
                meta[col] = row[col]

            source_val = None
            if source_col and source_col in df.columns:
                source_val = row[source_col]

            rows.append(
                {
                    "timestamp": float(row[ts_col]),
                    "value": row["__value_norm__"],
                    "source": source_val,
                    "meta": meta,
                }
            )

        rows.sort(key=lambda r: r["timestamp"])

        log_debug(
            _logger,
            "Standardized sentiment",
            rows=len(rows),
            ts_min=rows[0]["timestamp"] if rows else None,
            ts_max=rows[-1]["timestamp"] if rows else None,
        )

        return rows

    @staticmethod
    def _normalize_list(values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        norm: List[Dict[str, Any]] = []
        for v in values:
            
            ts = float(v.get("timestamp", 0.0))
            val = v.get("value")
            source = v.get("source")
            meta = v.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {"meta": meta}
            norm.append(
                {
                    "timestamp": ts,
                    "value": val,
                    "source": source,
                    "meta": meta,
                }
            )
        return norm

    # ------------------------------------------------------------------
    # Incremental ingestion (optional, used by live pipelines)
    # ------------------------------------------------------------------
    def on_new_value(
        self,
        ts: float,
        value: Union[float, Dict[str, Any]],
        source: Optional[str] = None,
        **meta: Any,
    ) -> None:
        """
        Ingest a single new sentiment datapoint (live mode).
        """
        entry = {
            "timestamp": float(ts),
            "value": value,
            "source": source,
            "meta": meta,
        }
        # assume mostly monotone; keep it cheap
        if not self._values or ts >= self._values[-1]["timestamp"]:
            self._values.append(entry)
        else:
            # rare out-of-order insertion
            self._values.append(entry)
            self._values.sort(key=lambda r: r["timestamp"])

    # ------------------------------------------------------------------
    # SentimentHandler-compatible API
    # ------------------------------------------------------------------
    @property
    def symbol(self) -> str:
        return self._symbol

    def latest_score(self) -> Union[float, Dict[str, Any], None]:
        if not self._values:
            return None
        return self._values[-1]["value"]

    def get_snapshot(self, ts: float) -> Union[float, Dict[str, Any], None]:
        """
        Return the latest sentiment value whose timestamp ≤ ts.
        Anti-lookahead is enforced by the ≤ condition.
        """
        if not self._values:
            return None

        # find all entries with timestamp ≤ ts
        eligible = [v for v in self._values if v["timestamp"] <= ts]
        if not eligible:
            return None
        return eligible[-1]["value"]

    def window(self, ts: float, n: int) -> List[Union[float, Dict[str, Any]]]:
        """
        Return the most recent n sentiment values with timestamp ≤ ts.
        """
        if not self._values:
            return []

        eligible = [v for v in self._values if v["timestamp"] <= ts]
        if not eligible:
            return []

        tail = eligible[-n:]
        return [v["value"] for v in tail]

    def ready(self) -> bool:
        return bool(self._values)

    def last_timestamp(self) -> Optional[float]:
        if not self._values:
            return None
        return float(self._values[-1]["timestamp"])

    def flush_cache(self) -> None:
        self._values.clear()