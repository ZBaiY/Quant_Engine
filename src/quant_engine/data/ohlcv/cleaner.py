"""
OHLCV cleaning utilities (v4).

This module must remain a PURE transformation layer:
- No networking / requests
- No filesystem I/O
- No JSON config loading
- No validation/reporting "checker" logic

It is allowed to:
- coerce schema
- cast dtypes
- handle missing
- (optionally) resample with explicit user intent
- (optionally) apply outlier handling with explicit user intent
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime as dt
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pytz


DEFAULT_REQUIRED_COLUMNS: Sequence[str] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
)


@dataclass(frozen=True)
class OHLCVCleanConfig:
    """
    Explicit cleaning config (v4).

    Keep it explicit and narrow. No implicit behavior.
    """
    required_columns: Sequence[str] = DEFAULT_REQUIRED_COLUMNS

    # core deterministic steps
    check_labels: bool = True
    cast_types: bool = True
    handle_missing: bool = True
    drop_duplicates: bool = True

    # optional / explicit steps (off by default)
    resample_align: bool = False
    resample_freq: str = "1h"  # pandas offset alias

    remove_outliers: bool = False
    outlier_threshold: float = 20.0

    substitute_outliers: bool = False
    adjacent_count: int = 5

    timezone_convert: bool = False
    utc_offset_hours: int | None = None  # e.g. +3 for UTC+3


class OHLCVCleaner:
    """
    OHLCV cleaner with v3-compatible implementations, v4-compatible boundaries.

    Important:
    - This class does NOT fetch or save.
    - This class does NOT perform "sanity checking reports".
      Tests should enforce expectations.
    """

    def __init__(self, config: OHLCVCleanConfig = OHLCVCleanConfig()) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # v3-ported operations (kept mostly intact)
    # ------------------------------------------------------------------

    def normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = list(self.cfg.required_columns)
        if "open_time" in df.columns:
            return df.reindex(columns=cols)
        # open_time is index
        cols_wo_index = [c for c in cols if c != "open_time"]
        return df.reindex(columns=cols_wo_index)

    def _coerce_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Binance returns ms timestamps; in parquet replay we might already have tz-aware datetimes.
        for col in ("open_time", "close_time"):
            if col not in df.columns:
                continue
            if np.issubdtype(df[col].dtype, np.datetime64):
                # ensure UTC
                df[col] = pd.to_datetime(df[col], utc=True)
            else:
                # assume ms epoch
                df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce", utc=True)
        return df

    def cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._coerce_time_columns(df)

        dtype_map: Mapping[str, str] = {
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "volume": "float32",
            "quote_asset_volume": "float32",
            "number_of_trades": "int32",
            "taker_buy_base_asset_volume": "float32",
            "taker_buy_quote_asset_volume": "float32",
            "ignore": "float32",
        }
        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        return df

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.ffill().bfill()

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        # Prefer deterministic de-dup by timestamp if present.
        if "open_time" in df.columns:
            df = df.sort_values("open_time").drop_duplicates(subset=["open_time"])
        return df.drop_duplicates()

    def resample_align(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Explicit resample. Off by default.

        Note: resampling is a policy decision; only enable when your ingestion pipeline asks for it.
        """
        if "open_time" not in df.columns:
            raise KeyError("resample_align requires 'open_time' column")

        df = df.copy()
        df = self._coerce_time_columns(df)
        df = df.set_index("open_time")

        agg_funcs = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "close_time": "last",
            "quote_asset_volume": "sum",
            "number_of_trades": "sum",
            "taker_buy_base_asset_volume": "sum",
            "taker_buy_quote_asset_volume": "sum",
            "ignore": "sum",
        }
        agg_funcs = {c: f for c, f in agg_funcs.items() if c in df.columns}

        out = (
            df.resample(self.cfg.resample_freq)
            .agg(agg_funcs)
            .ffill()
            .bfill()
            .drop_duplicates()
            .reset_index()
        )

        # enforce required columns order
        out = out.reindex(columns=list(self.cfg.required_columns), fill_value=0.0).copy()
        for col in self.cfg.required_columns:
            if col not in out.columns:
                out[col] = pd.NaT if "time" in col else (0 if col == "number_of_trades" else 0.0)
        return out

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score outlier removal (explicit).
        """
        threshold = float(self.cfg.outlier_threshold)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return df
        z = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0).replace(0, np.nan))
        mask = (z < threshold).all(axis=1)
        return df.loc[mask]

    def substitute_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score outlier substitution (explicit).
        """
        threshold = float(self.cfg.outlier_threshold)
        adjacent_count = int(self.cfg.adjacent_count)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return df

        z = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0).replace(0, np.nan))
        outliers = z > threshold
        half = adjacent_count // 2

        df = df.copy()
        for col in numeric_df.columns:
            idxs = outliers.index[outliers[col].fillna(False)]
            for i in idxs:
                pos = df.index.get_loc(i)
                start = max(0, pos - half)
                end = min(len(df), pos + half + 1)
                window = numeric_df[col].iloc[start:end].drop(i)
                if not window.empty:
                    df.at[i, col] = float(window.mean())
        return df

    def timezone_convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Explicit timezone conversion for time columns.
        Off by default.
        """
        utc_offset = self.cfg.utc_offset_hours
        df = df.copy()
        target_tz = dt.now().astimezone().tzinfo if utc_offset is None else pytz.FixedOffset(int(utc_offset) * 60)

        for col in df.columns:
            if "time" not in col:
                continue
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(target_tz)
        return df

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Canonical pipeline with explicit switches.
        Returns a DataFrame (same type as input), with canonical schema/dtypes.
        """
        df = self.normalize_schema(df)

        if self.cfg.check_labels:
            missing = [c for c in self.cfg.required_columns if (c != "open_time" and c not in df.columns)]
            if missing:
                raise ValueError(f"Missing required OHLCV columns: {missing}")

        if self.cfg.handle_missing:
            df = self.handle_missing(df)

        if self.cfg.cast_types:
            df = self.cast_types(df)

        if self.cfg.drop_duplicates:
            df = self.drop_duplicates(df)

        if self.cfg.remove_outliers:
            df = self.remove_outliers(df)

        if self.cfg.substitute_outliers:
            df = self.substitute_outliers(df)

        if self.cfg.resample_align:
            df = self.resample_align(df)

        if self.cfg.timezone_convert:
            df = self.timezone_convert(df)

        return df


def clean_ohlcv(df: pd.DataFrame, config: OHLCVCleanConfig = OHLCVCleanConfig()) -> pd.DataFrame:
    """
    Functional convenience wrapper.
    """
    return OHLCVCleaner(config).clean(df)