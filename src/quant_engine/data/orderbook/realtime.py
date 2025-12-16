from __future__ import annotations

import time
from typing import Any
from quant_engine.data.protocol_realtime import RealTimeDataHandler
from quant_engine.data.protocol_historical import HistoricalSignalSource
from quant_engine.utils.logger import get_logger, log_debug, log_info

from quant_engine.data.orderbook.cache import OrderbookCache
from quant_engine.data.orderbook.snapshot import OrderbookSnapshot


class RealTimeOrderbookHandler(RealTimeDataHandler):
    """Runtime orderbook handler (mode-agnostic).

    Conforms to runtime handler protocol semantics:
      - kwargs-driven __init__ (loader passes nested handler config via **cfg)
      - bootstrap(end_ts, lookback) present (no-op by default; IO-free runtime)
      - warmup_to(ts) establishes anti-lookahead anchor
      - get_snapshot(ts=None) / window(ts=None,n) are timestamp-aligned
      - BACKTEST seeding via from_historical(...) + driver replay into on_new_tick(...)

    Note: This handler stores OrderbookSnapshot objects (not DataFrames).
    """

    # --- declared attributes (protocol/typing) ---
    symbol: str
    source: str
    interval: str | None
    bootstrap_cfg: dict[str, Any]
    cache_cfg: dict[str, Any]
    cache: OrderbookCache
    _anchor_ts: float | None
    _logger: Any

    def __init__(self, symbol: str, **kwargs: Any):
        self.symbol = symbol

        # Optional metadata/routing
        source = kwargs.get("source", "binance")
        if not isinstance(source, str) or not source:
            raise ValueError("Orderbook 'source' must be a non-empty string")
        self.source = source

        ri = kwargs.get("interval")
        if ri is not None and (not isinstance(ri, str) or not ri):
            raise ValueError("Orderbook 'interval' must be a non-empty string if provided")
        self.interval = ri

        # Optional nested configs
        bootstrap = kwargs.get("bootstrap") or {}
        if not isinstance(bootstrap, dict):
            raise TypeError("Orderbook 'bootstrap' must be a dict")
        self.bootstrap_cfg = dict(bootstrap)

        cache = kwargs.get("cache") or {}
        if not isinstance(cache, dict):
            raise TypeError("Orderbook 'cache' must be a dict")
        self.cache_cfg = dict(cache)

        # cache depth precedence:
        #   1) cache.max_snaps
        #   2) legacy window
        #   3) default
        max_snaps = self.cache_cfg.get("max_snaps")
        if max_snaps is None:
            max_snaps = kwargs.get("window", 200)
        max_snaps_i = int(max_snaps)
        if max_snaps_i <= 0:
            raise ValueError("Orderbook cache.max_snaps must be > 0")

        self.cache = OrderbookCache(window=max_snaps_i)
        self._logger = get_logger(__name__)
        self._anchor_ts = None

        log_debug(
            self._logger,
            "RealTimeOrderbookHandler initialized",
            symbol=self.symbol,
            source=self.source,
            interval=self.interval,
            max_snaps=max_snaps_i,
            bootstrap=self.bootstrap_cfg,
        )

    # ------------------------------------------------------------------
    # Seeding (backtest)
    # ------------------------------------------------------------------

    @classmethod
    def from_historical(
        cls,
        historical_handler: HistoricalSignalSource,
        *,
        start_ts: float | None = None,
        window: int = 200,
        **kwargs: Any,
    ) -> "RealTimeOrderbookHandler":
        """Seed a runtime orderbook handler from a historical signal source.

        Anti-lookahead: if start_ts is provided, only items with ts <= start_ts are used.
        """
        rt = cls(symbol=historical_handler.symbol, window=window)

        # ---- config shadow (attach to rt, not self) ----
        source = kwargs.get("source", "binance")
        if not isinstance(source, str) or not source:
            raise ValueError("Orderbook 'source' must be a non-empty string")
        rt.source = source

        # allow either interval or refresh_interval (pick one convention, but tolerate both)
        ri = kwargs.get("interval", kwargs.get("refresh_interval"))
        if ri is not None and (not isinstance(ri, str) or not ri):
            raise ValueError("Orderbook 'interval' must be a non-empty string if provided")
        rt.interval = ri

        bootstrap = kwargs.get("bootstrap") or {}
        if not isinstance(bootstrap, dict):
            raise TypeError("Orderbook 'bootstrap' must be a dict")
        rt.bootstrap_cfg = dict(bootstrap)

        cache = kwargs.get("cache") or {}
        if not isinstance(cache, dict):
            raise TypeError("Orderbook 'cache' must be a dict")
        rt.cache_cfg = dict(cache)

        # cache depth precedence:
        #   1) cache.max_snaps
        #   2) legacy window
        #   3) default
        max_snaps = rt.cache_cfg.get("max_snaps")
        if max_snaps is None:
            max_snaps = kwargs.get("window", window)
        max_snaps_i = int(max_snaps)
        if max_snaps_i <= 0:
            raise ValueError("Orderbook cache.max_snaps must be > 0")

        rt.cache = OrderbookCache(window=max_snaps_i)
        rt._logger = get_logger(__name__)

        # ---- seed from historical (anti-lookahead handled by HistoricalSignalSource.window) ----
        seed = historical_handler.window(ts=start_ts, n=window)
        if not seed:
            rt._anchor_ts = start_ts
            return rt

        try:
            for item in seed:
                snap = _coerce_snapshot(rt.symbol, item)
                if snap is not None:
                    rt.cache.update(snap)
        except TypeError:
            snap = _coerce_snapshot(rt.symbol, seed)
            if snap is not None:
                rt.cache.update(snap)

        rt._anchor_ts = start_ts
        return rt
    # ------------------------------------------------------------------
    # Lifecycle (realtime/mock)
    # ------------------------------------------------------------------

    def bootstrap(self, *, end_ts: float, lookback: Any | None = None) -> None:
        """Preload recent data into cache.

        IO-free by default (no-op). Keeps params for observability/future adapters.
        """
        if lookback is None:
            lookback = self.bootstrap_cfg.get("lookback")

        log_debug(
            self._logger,
            "RealTimeOrderbookHandler.bootstrap (no-op)",
            symbol=self.symbol,
            source=self.source,
            end_ts=end_ts,
            lookback=lookback,
        )

    def warmup_to(self, ts: float) -> None:
        """Clamp implicit reads to ts (anti-lookahead anchor)."""
        self._anchor_ts = float(ts)
        log_debug(self._logger, "RealTimeOrderbookHandler warmup_to", symbol=self.symbol, anchor_ts=self._anchor_ts)

    # ------------------------------------------------------------------
    # Streaming tick API
    # ------------------------------------------------------------------

    def on_new_tick(self, snapshot: OrderbookSnapshot) -> None:
        """Protocol-compatible tick ingestion (live or replay)."""
        self.on_new_snapshot(snapshot)

    def on_new_snapshot(self, snapshot: OrderbookSnapshot) -> list[OrderbookSnapshot]:
        """Push a new orderbook snapshot into cache."""
        log_debug(self._logger, "RealTimeOrderbookHandler received snapshot", symbol=self.symbol)
        self.cache.update(snapshot)
        return self.cache.get_window()

    # ------------------------------------------------------------------
    # Unified access (timestamp-aligned)
    # ------------------------------------------------------------------

    def last_timestamp(self) -> float | None:
        snap = self.cache.latest()
        if snap is None:
            return None
        return float(snap.timestamp)

    def get_snapshot(self, ts: float | None = None) -> OrderbookSnapshot | None:
        """Return the latest OrderbookSnapshot aligned to ts (anti-lookahead)."""
        if ts is None:
            ts = self._anchor_ts if self._anchor_ts is not None else self.last_timestamp()
            if ts is None:
                return None
        snap = self.cache.latest_before_ts(float(ts))
        return snap

    def window(self, ts: float | None = None, n: int = 1) -> list[OrderbookSnapshot]:
        """Return last n snapshots aligned to ts (anti-lookahead)."""
        if ts is None:
            ts = self._anchor_ts if self._anchor_ts is not None else self.last_timestamp()
            if ts is None:
                return []
        return self.cache.window_before_ts(float(ts), int(n))

    # ------------------------------------------------------------------
    # Admin / tests
    # ------------------------------------------------------------------

    def reset(self) -> None:
        log_info(self._logger, "RealTimeOrderbookHandler reset requested", symbol=self.symbol)
        self.cache.clear()

    def run_mock(self, df, delay: float = 0.0):
        """v4-compliant simulated orderbook stream."""
        log_info(
            self._logger,
            "RealTimeOrderbookHandler starting mock stream",
            symbol=self.symbol,
            rows=len(df),
            delay=delay,
        )

        for _, row in df.iterrows():
            raw = row.to_dict()

            snapshot = OrderbookSnapshot(
                symbol=self.symbol,
                timestamp=float(raw["timestamp"]),
                best_bid=float(raw["best_bid"]),
                best_bid_size=float(raw["best_bid_size"]),
                best_ask=float(raw["best_ask"]),
                best_ask_size=float(raw["best_ask_size"]),
                bids=raw.get("bids", []),
                asks=raw.get("asks", []),
                latency=0.0,
            )

            window = self.on_new_snapshot(snapshot)
            yield snapshot, window

            if delay > 0:
                time.sleep(delay)


def _coerce_snapshot(symbol: str, x: Any) -> OrderbookSnapshot | None:
    if x is None:
        return None
    if isinstance(x, OrderbookSnapshot):
        return x
    if isinstance(x, dict):
        # tolerate alternative keys
        ts = x.get("timestamp", x.get("ts"))
        if ts is None:
            return None
        return OrderbookSnapshot(
            symbol=symbol,
            timestamp=float(ts),
            best_bid=float(x.get("best_bid", 0.0)),
            best_bid_size=float(x.get("best_bid_size", x.get("bid_size", 0.0))),
            best_ask=float(x.get("best_ask", 0.0)),
            best_ask_size=float(x.get("best_ask_size", x.get("ask_size", 0.0))),
            bids=x.get("bids", []),
            asks=x.get("asks", []),
            latency=float(x.get("latency", 0.0)),
        )
    return None
