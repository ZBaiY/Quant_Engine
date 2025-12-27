from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Literal, cast

import datetime as _dt
from pathlib import Path

import pandas as pd
import requests


@dataclass(frozen=True)
class BinanceOHLCVFetcherConfig:
    base_url: str = "https://api.binance.com"
    endpoint: str = "/api/v3/klines"
    timeout_s: float = 30.0
    max_limit: int = 1000  # Binance klines limit is up to 1000

@dataclass(frozen=True)
class BinanceOHLCVCleanerConfig:
    """Cleaner config.

    This cleaner is intentionally conservative:
      - no resampling
      - no gap filling
      - no outlier logic

    It enforces contracts + produces a report.
    """

    interval: str
    strict: bool = True
    dropna: bool = True
    drop_zero_price_bars: bool = True
    allow_empty: bool = True

@dataclass(frozen=True)
class BinanceOHLCVBackfillConfig:
    symbol: str
    interval: str
    start_ms: int | float | str | _dt.datetime | pd.Timestamp
    end_ms: int | float | str | _dt.datetime | pd.Timestamp

    limit: int | None = None

    # write switches
    write_raw: bool = True
    write_cleaned: bool = True

    # cleaner semantics
    strict: bool = False  # for historical backfill, default is non-strict (gaps are common)
    dropna: bool = True
    drop_zero_price_bars: bool = True


@dataclass(frozen=True)
class OHLCVCleanReport:
    n_in: int
    n_out: int
    dup_dropped: int
    rows_dropped_na: int
    zero_price_bars_dropped: int
    gaps: int
    gap_sizes_ms: list[int]
    first_data_ts: int | None
    last_data_ts: int | None


    def summary(self) -> str:
        lines = [
            f"OHLCV Clean Report:",
            f"  Input rows: {self.n_in}",
            f"  Output rows: {self.n_out}",
            f"  Duplicates dropped: {self.dup_dropped}",
            f"  Rows dropped (NA): {self.rows_dropped_na}",
            f"  Zero-price bars dropped: {self.zero_price_bars_dropped}",
            f"  Gaps detected: {self.gaps}",
        ]
        if self.gaps > 0:
            lines.append(f"    Gap sizes (ms): {self.gap_sizes_ms[:10]}{'...' if len(self.gap_sizes_ms) > 10 else ''}")
        lines.append(f"  First data_ts: {self.first_data_ts}")
        lines.append(f"  Last data_ts: {self.last_data_ts}")
        return "\n".join(lines)


class BinanceOHLCVFetcher:
    """Fetch Binance klines and normalize to v4 time semantics.

    Output schema (raw-normalized):
      - data_ts: int64 epoch ms (BAR CLOSE time)  <-- authoritative event time
      - open_time: int64 epoch ms (BAR OPEN time)
      - close_time: datetime64[ns, UTC] (inspection only)
      - core: open, high, low, close, volume
      - aux: quote_asset_volume, number_of_trades, taker_buy_*, ignore

    Notes:
      - This fetcher is intentionally *pure I/O + normalization*.
      - Cleaning (dedup rules, gap repair, resample, outliers) belongs to a separate cleaner.
    """

    def __init__(self, *, cfg: BinanceOHLCVFetcherConfig | None = None, session: requests.Session | None = None):
        self._cfg = cfg or BinanceOHLCVFetcherConfig()
        self._session = session or requests.Session()

    def _url(self) -> str:
        return f"{self._cfg.base_url}{self._cfg.endpoint}"

    def fetch_chunk(
        self,
        *,
        symbol: str,
        interval: str,
        start_ms: int | float | str | _dt.datetime | pd.Timestamp,
        end_ms: int | float | str | _dt.datetime | pd.Timestamp | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch a single kline chunk.

        start_ms/end_ms are coerced to epoch-ms ints.
        Returns an *empty* DataFrame if the API returns no rows.
        """
        start_ms_i = _coerce_epoch_ms(start_ms)
        end_ms_i = _coerce_epoch_ms(end_ms) if end_ms is not None else None

        lim = int(limit or self._cfg.max_limit)
        if lim <= 0 or lim > self._cfg.max_limit:
            raise ValueError(f"limit must be in [1, {self._cfg.max_limit}]")

        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms_i,
            "limit": lim,
        }
        if end_ms_i is not None:
            params["endTime"] = end_ms_i

        r = self._session.get(self._url(), params=params, timeout=self._cfg.timeout_s)
        r.raise_for_status()
        payload = r.json()
        if not payload:
            return pd.DataFrame()

        cols = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "_close_time_ms",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
        df = pd.DataFrame(payload, columns=cols)

        # ---- enforce dtypes early (avoid CSV-era dtype drift) ----
        df["open_time"] = df["open_time"].astype("int64")
        df["_close_time_ms"] = df["_close_time_ms"].astype("int64")
        df["number_of_trades"] = df["number_of_trades"].astype("int64")

        for c in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ):
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # ---- v4 time semantics ----
        df["data_ts"] = df["_close_time_ms"].astype("int64")
        df["close_time"] = pd.to_datetime(df["_close_time_ms"], unit="ms", utc=True)
        df = df.drop(columns=["_close_time_ms"])

        # Order columns: time first.
        front = ["data_ts", "open_time", "close_time"]
        rest = [c for c in df.columns if c not in front]
        df = df[front + rest]

        return df

    def iter_range(
        self,
        *,
        symbol: str,
        interval: str,
        start_ms: int | float | str | _dt.datetime | pd.Timestamp,
        end_ms: int | float | str | _dt.datetime | pd.Timestamp,
        limit: int | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Yield successive chunks covering [start_ms, end_ms) by advancing on open_time."""
        step = _interval_ms(interval)
        cur = _coerce_epoch_ms(start_ms)
        end = _coerce_epoch_ms(end_ms)

        while cur < end:
            df = self.fetch_chunk(symbol=symbol, interval=interval, start_ms=cur, end_ms=end, limit=limit)
            if df.empty:
                return
            yield df

            # Advance by last open_time + interval. Binance returns open_time aligned to interval.
            last_open = int(df["open_time"].iloc[-1])
            nxt = last_open + step
            if nxt <= cur:
                # Defensive: avoid infinite loop if payload is pathological.
                return
            cur = nxt

    def fetch_range(
        self,
        *,
        symbol: str,
        interval: str,
        start_ms: int | float | str | _dt.datetime | pd.Timestamp,
        end_ms: int | float | str | _dt.datetime | pd.Timestamp,
        limit: int | None = None,
        dedup: bool = True,
    ) -> pd.DataFrame:
        """Fetch a full range into one DataFrame.

        dedup=True will drop duplicates by `data_ts` and sort ascending.
        """
        parts: list[pd.DataFrame] = []
        for chunk in self.iter_range(symbol=symbol, interval=interval, start_ms=start_ms, end_ms=end_ms, limit=limit):
            parts.append(chunk)

        if not parts:
            return pd.DataFrame()

        df = pd.concat(parts, ignore_index=True)
        if dedup and "data_ts" in df.columns:
            df = df.drop_duplicates(subset=["data_ts"], keep="last")
        df = df.sort_values("data_ts", kind="stable").reset_index(drop=True)
        return df


def concat_chunks(chunks: Iterable[pd.DataFrame], *, dedup: bool = True) -> pd.DataFrame:
    """Utility: concat already-fetched chunks (keeps time semantics)."""
    parts = [c for c in chunks if c is not None and not c.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if dedup and "data_ts" in df.columns:
        df = df.drop_duplicates(subset=["data_ts"], keep="last")
    if "data_ts" in df.columns:
        df = df.sort_values("data_ts", kind="stable").reset_index(drop=True)
    return df


class BinanceOHLCVCleaner:
    """Clean + validate normalized Binance OHLCV frames.

    Expected input columns (at minimum):
      - data_ts (int ms)
      - open_time (int ms)
      - open, high, low, close, volume

    Optional:
      - close_time (datetime-like; inspection only)
      - aux columns

    Guarantees on output:
      - sorted by data_ts ascending (stable)
      - duplicates dropped by data_ts (keep last)
      - data_ts/open_time are int64
      - numeric core columns are numeric (float64)
    """

    _CORE_NUM = ("open", "high", "low", "close", "volume")
    _REQ = ("data_ts", "open_time", *_CORE_NUM)

    def __init__(self, *, cfg: BinanceOHLCVCleanerConfig):
        self._cfg = cfg
        self._step = _interval_ms(cfg.interval)

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, OHLCVCleanReport]:
        if df is None or len(df) == 0:
            if self._cfg.allow_empty:
                rep = OHLCVCleanReport(
                    n_in=0,
                    n_out=0,
                    dup_dropped=0,
                    rows_dropped_na=0,
                    zero_price_bars_dropped=0,
                    gaps=0,
                    gap_sizes_ms=[],
                    first_data_ts=None,
                    last_data_ts=None,
                )
                return pd.DataFrame(), rep
            raise ValueError("Empty OHLCV frame")

        missing = [c for c in self._REQ if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        n_in = int(len(df))
        x = df.copy()

        # --- dtypes ---
        x["data_ts"] = pd.to_numeric(x["data_ts"], errors="coerce").astype("Int64")
        x["open_time"] = pd.to_numeric(x["open_time"], errors="coerce").astype("Int64")

        for c in self._CORE_NUM:
            x[c] = pd.to_numeric(x[c], errors="coerce")

        if "number_of_trades" in x.columns:
            x["number_of_trades"] = pd.to_numeric(x["number_of_trades"], errors="coerce")

        if "close_time" in x.columns and not pd.api.types.is_datetime64_any_dtype(x["close_time"]):
            # Accept either datetime-like strings or ms-int.
            try:
                if pd.api.types.is_integer_dtype(x["close_time"]):
                    x["close_time"] = pd.to_datetime(x["close_time"], unit="ms", utc=True)
                else:
                    x["close_time"] = pd.to_datetime(x["close_time"], utc=True)
            except Exception:
                # keep as-is (inspection-only); strict mode may fail later if user expects it.
                pass

        # --- NA drop ---
        rows_dropped_na = 0
        if self._cfg.dropna:
            before = len(x)
            x = x.dropna(subset=["data_ts", "open_time", *self._CORE_NUM])
            rows_dropped_na = before - len(x)

        # --- zero-price bar drop (Binance maintenance/glitch signature) ---
        zero_price_bars_dropped = 0
        if self._cfg.drop_zero_price_bars:
            before = len(x)
            price_cols = ["open", "high", "low", "close"]
            is_zero_bar = (x[price_cols] == 0).all(axis=1)
            x = x.loc[~is_zero_bar].copy()
            zero_price_bars_dropped = before - len(x)

        # Convert Int64 (nullable) -> int64
        try:
            x["data_ts"] = x["data_ts"].astype("int64")
            x["open_time"] = x["open_time"].astype("int64")
        except Exception as e:
            raise TypeError("data_ts/open_time must be coercible to int64") from e

        # --- dedup + sort ---
        x = x.sort_values("data_ts", kind="stable")
        before = len(x)
        x = x.drop_duplicates(subset=["data_ts"], keep="last")
        dup_dropped = before - len(x)
        x = x.reset_index(drop=True)

        # --- gap detection (based on close-time cadence) ---
        gap_sizes: list[int] = []
        gaps = 0
        if len(x) >= 2:
            dt = x["data_ts"].diff().iloc[1:]
            bad = dt[dt != self._step]
            # treat negative/zero as gap as well (ordering bugs)
            bad = bad[bad.notna()]
            gaps = int(len(bad))
            gap_sizes = [int(v) for v in bad.to_list()]

        if self._cfg.strict:
            # Binance invariant: data_ts increases by step.
            if gaps != 0:
                raise ValueError(f"Detected {gaps} cadence gaps (ms diffs != {self._step}): {gap_sizes[:10]}")
            # Basic OHLC sanity.
            if (x["high"] < x[["open", "close"]].max(axis=1)).any() or (x["low"] > x[["open", "close"]].min(axis=1)).any():
                raise ValueError("OHLC invariant violated: high/low inconsistent with open/close")
            if (x[["open", "high", "low", "close"]] <= 0).any().any():
                raise ValueError("Non-positive prices detected after cleaning")

        n_out = int(len(x))
        rep = OHLCVCleanReport(
            n_in=n_in,
            n_out=n_out,
            dup_dropped=int(dup_dropped),
            rows_dropped_na=int(rows_dropped_na),
            zero_price_bars_dropped=int(zero_price_bars_dropped),
            gaps=int(gaps),
            gap_sizes_ms=gap_sizes,
            first_data_ts=int(x["data_ts"].iloc[0]) if n_out else None,
            last_data_ts=int(x["data_ts"].iloc[-1]) if n_out else None,
        )
        return x, rep




@dataclass(frozen=True)
class OHLCVParquetStoreConfig:
    """Filesystem layout for chunked parquet writes.

    Layout (day):
      <root>/<stage>/OHLCV/BINANCE/<symbol>/<interval>/<YYYY>/<MM>/<DD>.parquet

    Layout (year):
      <root>/<stage>/OHLCV/BINANCE/<symbol>/<interval>/<YYYY>.parquet

    Notes:
      - day layout is the safe default for streaming/chunk writes.
      - year layout is convenient for reads, but appending requires read+rewrite (expensive).
    """

    root: str = "data"
    domain: str = "OHLCV"
    source: str = "BINANCE"
    layout: Literal["day", "year"] = "day"


class OHLCVParquetStore:
    def __init__(self, *, cfg: OHLCVParquetStoreConfig | None = None):
        self._cfg = cfg or OHLCVParquetStoreConfig()

    def _base_dir(self, *, stage: Literal["raw", "cleaned"], symbol: str, interval: str) -> Path:
        return Path(self._cfg.root) / stage / self._cfg.domain / self._cfg.source / symbol / interval

    def _path_for(self, *, stage: Literal["raw", "cleaned"], symbol: str, interval: str, year: int, month: int | None = None, day: int | None = None) -> Path:
        base = self._base_dir(stage=stage, symbol=symbol, interval=interval)
        if self._cfg.layout == "year":
            return base / f"{year:04d}.parquet"
        if month is None or day is None:
            raise ValueError("month/day required for day layout")
        return base / f"{year:04d}" / f"{month:02d}" / f"{day:02d}.parquet"

    def _append_dedup_write(self, *, df: pd.DataFrame, path: Path, dedup_key: str = "data_ts") -> None:
        """Append by read+concat+dedup+rewrite.

        This is cheap for day files and expensive for year files.
        """
        _ensure_dir(path.parent)
        if path.exists():
            old = pd.read_parquet(path)
            merged = pd.concat([old, df], ignore_index=True)
        else:
            merged = df

        if dedup_key in merged.columns:
            merged = merged.drop_duplicates(subset=[dedup_key], keep="last")
            merged = merged.sort_values(dedup_key, kind="stable").reset_index(drop=True)

        merged.to_parquet(path, index=False)

    def write_chunk(
        self,
        df: pd.DataFrame,
        *,
        stage: Literal["raw", "cleaned"],
        symbol: str,
        interval: str,
    ) -> list[Path]:
        """Write a (possibly cross-day) chunk to parquet according to layout.

        Returns the list of written file paths.
        """
        if df is None or df.empty:
            return []
        if "data_ts" not in df.columns:
            raise ValueError("write_chunk requires data_ts column")

        ts = pd.to_datetime(df["data_ts"], unit="ms", utc=True)
        x = df.copy()
        x["_year"] = ts.dt.year.astype("int32")
        x["_month"] = ts.dt.month.astype("int16")
        x["_day"] = ts.dt.day.astype("int16")

        written: list[Path] = []

        if self._cfg.layout == "year":
            for y, sub in x.groupby("_year", sort=True):
                out = self._path_for(stage=stage, symbol=symbol, interval=interval, year=_as_int(y))
                sub2 = sub.drop(columns=["_year", "_month", "_day"], errors="ignore")
                self._append_dedup_write(df=sub2, path=out)
                written.append(out)
            return written

        # day layout
        for (y, m, d), sub in x.groupby(["_year", "_month", "_day"], sort=True):
            out = self._path_for(
                stage=stage,
                symbol=symbol,
                interval=interval,
                year=_as_int(y),
                month=_as_int(m),
                day=_as_int(d),
            )
            sub2 = sub.drop(columns=["_year", "_month", "_day"], errors="ignore")
            self._append_dedup_write(df=sub2, path=out)
            written.append(out)

        return written




class BinanceOHLCVBackfiller:
    """Chunked OHLCV backfill: fetch -> (optional clean) -> write parquet.

    - Fetcher owns cursor progression (open_time + interval) to avoid cleaned-induced drift.
    - Cleaner never fills gaps; it may drop invalid rows and reports drops.
    - Store writes in day/year layout with dedup on data_ts.
    """

    def __init__(
        self,
        *,
        fetcher: BinanceOHLCVFetcher | None = None,
        store: OHLCVParquetStore | None = None,
    ):
        self._fetcher = fetcher or BinanceOHLCVFetcher()
        self._store = store or OHLCVParquetStore()

    def run(self, *, cfg: BinanceOHLCVBackfillConfig) -> dict[str, Any]:
        cleaner = BinanceOHLCVCleaner(
            cfg=BinanceOHLCVCleanerConfig(
                interval=cfg.interval,
                strict=cfg.strict,
                dropna=cfg.dropna,
                drop_zero_price_bars=cfg.drop_zero_price_bars,
                allow_empty=True,
            )
        )

        totals = {
            "chunks": 0,
            "raw_rows": 0,
            "clean_rows": 0,
            "rows_dropped_na": 0,
            "zero_price_bars_dropped": 0,
            "dup_dropped_in_cleaner": 0,
            "written_raw_files": 0,
            "written_cleaned_files": 0,
        }

        for chunk in self._fetcher.iter_range(
            symbol=cfg.symbol,
            interval=cfg.interval,
            start_ms=cfg.start_ms,
            end_ms=cfg.end_ms,
            limit=cfg.limit,
        ):
            totals["chunks"] += 1
            totals["raw_rows"] += int(len(chunk))

            if cfg.write_raw:
                written = self._store.write_chunk(chunk, stage="raw", symbol=cfg.symbol, interval=cfg.interval)
                totals["written_raw_files"] += len(written)

            if cfg.write_cleaned:
                cleaned, rep = cleaner.clean(chunk)
                totals["clean_rows"] += int(len(cleaned))
                totals["rows_dropped_na"] += int(rep.rows_dropped_na)
                totals["zero_price_bars_dropped"] += int(rep.zero_price_bars_dropped)
                totals["dup_dropped_in_cleaner"] += int(rep.dup_dropped)

                written = self._store.write_chunk(cleaned, stage="cleaned", symbol=cfg.symbol, interval=cfg.interval)
                totals["written_cleaned_files"] += len(written)

        return totals
    


def _interval_ms(interval: str) -> int:
    """Binance interval string -> milliseconds."""
    if not interval or not isinstance(interval, str):
        raise TypeError("interval must be a non-empty string")

    unit = interval[-1]
    try:
        n = int(interval[:-1])
    except Exception as e:
        raise ValueError(f"Invalid interval: {interval!r}") from e

    if n <= 0:
        raise ValueError(f"Invalid interval: {interval!r}")

    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 3_600_000
    if unit == "d":
        return n * 86_400_000
    if unit == "w":
        return n * 7 * 86_400_000

    # Binance also supports "M" (month) but it is not fixed-length in ms.
    if unit == "M":
        raise ValueError("Monthly interval 'M' is not supported for ms conversion")

    raise ValueError(f"Unsupported interval unit: {unit!r}")


def _coerce_epoch_ms(x: Any) -> int:
    """Coerce timestamp-like input into epoch milliseconds int."""
    if x is None:
        raise TypeError("timestamp cannot be None")

    if isinstance(x, bool):
        raise TypeError("timestamp cannot be bool")

    if isinstance(x, int):
        # Heuristic: if user passed seconds, upscale.
        return x * 1000 if x < 10_000_000_000 else x

    if isinstance(x, float):
        # If float seconds, upscale; if float ms, round.
        if x < 10_000_000_000:
            return int(round(x * 1000))
        return int(round(x))

    if isinstance(x, (pd.Timestamp, _dt.datetime)):
        ts = x
        if isinstance(ts, _dt.datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=_dt.timezone.utc)
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return int(ts.value // 1_000_000)
        return int(ts.timestamp() * 1000)

    if isinstance(x, str):
        ts = pd.to_datetime(x, utc=True)
        return int(ts.value // 1_000_000)

    raise TypeError(f"Unsupported timestamp type: {type(x).__name__}")

# --------------------------------------------------------------------------------------
# Parquet storage + backfill (chunked)
# --------------------------------------------------------------------------------------



def _as_int(x: Any) -> int:
    """Coerce pandas/numpy scalars (and friends) to built-in int.

    Pylance is conservative about `int(object)`; we narrow/cast explicitly.
    """
    # Fast paths
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    if isinstance(x, (str, bytes, bytearray)):
        return int(x)

    # numpy/pandas scalar -> python scalar
    item = getattr(x, "item", None)
    if callable(item):
        try:
            v = item()
            # recurse once in case v is still a scalar
            if v is not x:
                return _as_int(v)
        except Exception:
            pass

    # Last resort: runtime conversion; cast for type checker.
    return int(cast("int | float | str | bytes | bytearray", x))

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
