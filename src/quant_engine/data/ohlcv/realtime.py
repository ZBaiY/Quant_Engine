from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np
from quant_engine.data.ohlcv.historical import HistoricalOHLCVHandler
from quant_engine.data.contracts.protocol_historical import HistoricalSignalSource
from quant_engine.data.contracts.protocol_realtime import TimestampLike, RealTimeDataHandler
from quant_engine.utils.logger import get_logger, log_debug, log_info

from .cache import DataCache
from .snapshot import OHLCVSnapshot


class OHLCVDataHandler(RealTimeDataHandler):
    """Runtime OHLCV handler (mode-agnostic).

    This handler is the *runtime platform* used in ALL modes.

    Config mapping (Strategy.DATA.*.ohlcv):
      - source: data origin identifier (default: "binance"). Runtime handler stores it for routing/logging.
      - interval: bar interval string (e.g., "1m", "15m"). Stored for validation/metadata.
      - bootstrap.lookback: convenience horizon for Engine.bootstrap(); handler may use it as default.
      - cache.max_bars: in-memory cache depth (ring buffer size).

    Important:
      - IO-free by default: no networking, no filesystem.
      - Backtest seeding: `from_historical(...)` then driver replays via `on_new_tick(...)`.
      - Anti-lookahead: reads clamp to `ts` (explicit) or `_anchor_ts` (set by warmup_to).
    """

    # --- declared attributes (protocol/typing) ---
    symbol: str
    source: str
    interval: str
    bootstrap_cfg: dict[str, Any]
    cache_cfg: dict[str, Any]
    cache: DataCache
    _anchor_ts: float | None
    _logger: Any

    def __init__(self, symbol: str, **kwargs: Any):
        """Runtime handler init.

        IMPORTANT: Keep init kwargs-driven.
        Strategy/Loader passes nested handler config via `**cfg`.
        """
        self.symbol = symbol

        # Required semantic fields
        interval = kwargs.get("interval")
        if not isinstance(interval, str) or not interval:
            raise ValueError("OHLCV handler requires non-empty 'interval' (e.g. '1m')")
        self.interval = interval

        # Optional metadata/routing
        source = kwargs.get("source", "binance")
        if not isinstance(source, str) or not source:
            raise ValueError("OHLCV 'source' must be a non-empty string")
        self.source = source

        # Optional nested configs
        bootstrap = kwargs.get("bootstrap") or {}
        if not isinstance(bootstrap, dict):
            raise TypeError("OHLCV 'bootstrap' must be a dict")
        self.bootstrap_cfg = dict(bootstrap)

        cache = kwargs.get("cache") or {}
        if not isinstance(cache, dict):
            raise TypeError("OHLCV 'cache' must be a dict")
        self.cache_cfg = dict(cache)

        # cache depth precedence:
        #   1) cache.max_bars
        #   2) legacy window
        #   3) default
        max_bars = self.cache_cfg.get("max_bars")
        if max_bars is None:
            max_bars = kwargs.get("window", 1000)
        max_bars_i = int(max_bars)
        if max_bars_i <= 0:
            raise ValueError("OHLCV cache.max_bars must be > 0")

        self.cache = DataCache(window=max_bars_i)

        self._logger = get_logger(__name__)
        self._anchor_ts = None

        log_debug(
            self._logger,
            "RealTimeDataHandler initialized",
            symbol=self.symbol,
            source=self.source,
            interval=self.interval,
            max_bars=max_bars_i,
            bootstrap=self.bootstrap_cfg,
        )


    # ------------------------------------------------------------------
    # Lifecycle (realtime/mock)
    # ------------------------------------------------------------------

    def bootstrap(self, *, end_ts: float, lookback: Any | None = None) -> None:
        """Preload recent data into cache.

        By default: no-op (IO-free handler).

        We still store bootstrap params for observability and to allow later adapters.
        """
        if lookback is None:
            lookback = self.bootstrap_cfg.get("lookback")

        log_debug(
            self._logger,
            "RealTimeDataHandler.bootstrap (no-op)",
            symbol=self.symbol,
            source=self.source,
            interval=self.interval,
            end_ts=end_ts,
            lookback=lookback,
        )

    def align_to(self, ts: float) -> None:
        """Clamp implicit reads to ts (anti-lookahead anchor)."""
        self._anchor_ts = float(ts)
        log_debug(
            self._logger,
            "RealTimeDataHandler align_to",
            symbol=self.symbol,
            anchor_ts=self._anchor_ts,
        )

    # ------------------------------------------------------------------
    # Streaming tick API
    # ------------------------------------------------------------------

    def on_new_tick(self, bar: Any) -> None:
        """Push one new bar into the cache (live or replay)."""
        df = _coerce_ohlcv_to_df(bar)
        if df is None or df.empty:
            log_debug(
                self._logger,
                "RealTimeDataHandler.on_new_tick: empty bar ignored",
                symbol=self.symbol,
            )
            return

        df = _ensure_timestamp(df)

        # If caller passed multiple rows, append deterministically.
        df = df.sort_values("timestamp")
        for _, row in df.iterrows():
            self.cache.update(row.to_frame().T)

    # ------------------------------------------------------------------
    # Unified access (timestamp-aligned)
    # ------------------------------------------------------------------

    def last_timestamp(self) -> float | None:
        df = self.cache.get_window()
        if df is None or df.empty or "timestamp" not in df.columns:
            return None
        try:
            return float(df["timestamp"].iloc[-1])
        except Exception:
            return None

    def get_snapshot(self, ts: float | None = None) -> OHLCVSnapshot | None:
        """Return the latest bar snapshot aligned to ts (anti-lookahead)."""
        if ts is None:
            ts = self._anchor_ts if self._anchor_ts is not None else self.last_timestamp()
            if ts is None:
                return None

        bar = self.cache.latest_before_ts(float(ts))
        if bar is None or bar.empty:
            return None

        row = bar.iloc[-1]
        return OHLCVSnapshot.from_bar(float(ts), row)

    def window(self, ts: float | None = None, n: int = 1) -> pd.DataFrame:
        """Return a DataFrame of the last n bars aligned to ts (anti-lookahead)."""
        if ts is None:
            ts = self._anchor_ts if self._anchor_ts is not None else self.last_timestamp()
            if ts is None:
                return pd.DataFrame()
        return self.cache.window_before_ts(float(ts), int(n))

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    def window_df(self, window: int | None = None) -> pd.DataFrame:
        """Deprecated. Prefer window(ts, n)."""
        df = self.cache.get_window() or pd.DataFrame()
        if window is not None and not df.empty:
            return df.tail(int(window))
        return df

    def reset(self) -> None:
        log_info(self._logger, "RealTimeDataHandler reset requested", symbol=self.symbol)
        try:
            self.cache.clear()
        except AttributeError:
            pass

    def run_mock(self, df: pd.DataFrame, delay: float = 1.0):
        """A mock real-time stream for testing without exchange."""
        import time

        log_info(
            self._logger,
            "RealTimeDataHandler starting mock stream",
            symbol=self.symbol,
            rows=len(df),
            delay=delay,
        )
        for _, row in df.iterrows():
            bar = row.to_frame().T
            self.on_new_tick(bar)
            yield bar, self.cache.get_window()
            time.sleep(delay)


def _coerce_ohlcv_to_df(x: Any) -> pd.DataFrame | None:
    """Coerce common bar payloads into a DataFrame."""
    if x is None:
        return None

    if isinstance(x, pd.DataFrame):
        df = x
    elif isinstance(x, dict):
        df = pd.DataFrame([x])
    else:
        try:
            df = pd.DataFrame(x)
        except Exception:
            return None

    if df.empty:
        return df

    # normalize common column aliases
    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "timestamp"})

    return df


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has float-seconds 'timestamp' column.

    Accepted inputs:
      - already has 'timestamp' (seconds)
      - has 'open_time' (ms epoch or datetime)
    """
    if "timestamp" in df.columns:
        # best-effort cast
        try:
            df = df.copy()
            df["timestamp"] = df["timestamp"].astype(float)
        except Exception:
            pass
        return df

    if "open_time" not in df.columns:
        raise KeyError("OHLCV bar must contain 'timestamp' or 'open_time'")

    out = df.copy()
    s = out["open_time"]

    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, utc=True, errors="coerce")      # Series[datetime64[ns, UTC]]
        ns = dt.astype("int64")                                # Series[int64] (ns since epoch; NaT -> min int)
        sec = ns.astype("float64") / 1_000_000_000.0
        out["timestamp"] = sec.where(dt.notna(), np.nan)        # 把 NaT 修回 NaN
        return out

    # assume ms epoch
    out["timestamp"] = (pd.to_numeric(s, errors="coerce") / 1000.0).astype(float)
    return out