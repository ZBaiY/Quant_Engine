from __future__ import annotations

from collections import deque
from typing import Any, Deque, Optional, Iterable

from quant_engine.utils.logger import get_logger, log_debug

from quant_engine.data.contracts.protocol_realtime import RealTimeDataHandler, TimestampLike
from quant_engine.data.contracts.protocol_historical import HistoricalSignalSource
from quant_engine.data.derivatives.option_chain.chain_handler import OptionChainDataHandler
from quant_engine.data.derivatives.option_chain.snapshot import OptionChainSnapshot
from quant_engine.data.derivatives.iv.snapshot import IVSurfaceSnapshot


class IVSurfaceDataHandler(RealTimeDataHandler):
    """Runtime IV surface handler (derived layer, mode-agnostic).

    Contract / protocol shadow:
      - kwargs-driven __init__ (loader/builder passes handler config via **cfg)
      - bootstrap(end_ts, lookback) present (no-op by default; IO-free)
      - warmup_to(ts) establishes anti-lookahead anchor
      - get_snapshot(ts=None) / window(ts=None,n) are timestamp-aligned
      - last_timestamp() supported

    Semantics:
      - This handler is a *derived* layer from an underlying OptionChainDataHandler.
      - It does not touch exchange APIs; it only converts OptionChainSnapshot -> IVSurfaceSnapshot.
      - A real SABR/SSVI calibrator can be plugged later without changing this API.

    Config (Strategy.DATA.*.iv_surface):
      - source: routing/metadata (default: "deribit")
      - interval: required cadence (e.g. "1m", "5m")
      - bootstrap.lookback: convenience horizon for Engine.bootstrap()
      - cache.max_bars: in-memory cache depth (IVSurfaceSnapshot)
      - expiry: optional expiry selector (future)
      - model_name: optional label (e.g. "SSVI", "SABR", "CHAIN_DERIVED")

    NOTE: `chain_handler` must be provided via kwargs at construction time by the builder.
    """

    # --- declared attributes (protocol/typing shadow) ---
    symbol: str
    chain_handler: OptionChainDataHandler
    source: str
    interval: str
    bootstrap_cfg: dict[str, Any]
    cache_cfg: dict[str, Any]
    expiry: str | None
    model_name: str

    _snapshots: Deque[IVSurfaceSnapshot]
    _anchor_ts: float | None
    _logger: Any

    def __init__(self, symbol: str, **kwargs: Any):
        self.symbol = symbol
        self._logger = get_logger(__name__)

        # required: chain_handler
        ch = kwargs.get("chain_handler") or kwargs.get("option_chain_handler")
        if not isinstance(ch, OptionChainDataHandler):
            raise ValueError("IVSurfaceDataHandler requires 'chain_handler' (OptionChainDataHandler) in kwargs")
        self.chain_handler = ch

        # required: interval
        ri = kwargs.get("interval")
        if not isinstance(ri, str) or not ri:
            raise ValueError("IV surface handler requires non-empty 'interval' (e.g. '5m')")
        self.interval = ri

        # optional metadata/routing
        src = kwargs.get("source", "deribit")
        if not isinstance(src, str) or not src:
            raise ValueError("IV surface 'source' must be a non-empty string")
        self.source = src

        # optional model/expiry
        expiry = kwargs.get("expiry")
        if expiry is not None and (not isinstance(expiry, str) or not expiry):
            raise ValueError("IV surface 'expiry' must be a non-empty string if provided")
        self.expiry = expiry

        model_name = kwargs.get("model_name", kwargs.get("model", "CHAIN_DERIVED"))
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("IV surface 'model_name' must be a non-empty string")
        self.model_name = model_name

        # optional nested configs
        bootstrap = kwargs.get("bootstrap") or {}
        if not isinstance(bootstrap, dict):
            raise TypeError("IV surface 'bootstrap' must be a dict")
        self.bootstrap_cfg = dict(bootstrap)

        cache = kwargs.get("cache") or {}
        if not isinstance(cache, dict):
            raise TypeError("IV surface 'cache' must be a dict")
        self.cache_cfg = dict(cache)

        max_bars = self.cache_cfg.get("max_bars")
        if max_bars is None:
            max_bars = kwargs.get("window", 1000)
        max_bars_i = int(max_bars)
        if max_bars_i <= 0:
            raise ValueError("IV surface cache.max_bars must be > 0")

        self._snapshots = deque(maxlen=max_bars_i)
        self._anchor_ts = None

        log_debug(
            self._logger,
            "IVSurfaceDataHandler initialized",
            symbol=self.symbol,
            source=self.source,
            interval=self.interval,
            max_bars=max_bars_i,
            bootstrap=self.bootstrap_cfg,
            model_name=self.model_name,
            expiry=self.expiry,
        )

    # ------------------------------------------------------------------
    # Seeding (backtest)
    # ------------------------------------------------------------------

    @classmethod
    def from_historical(
        cls,
        historical_handler: HistoricalSignalSource,
        *,
        start_ts: TimestampLike | None = None,
        window: int = 200,
        **kwargs: Any,
    ) -> "IVSurfaceDataHandler":
        """Seed a runtime IV surface handler from a historical signal source.

        Anti-lookahead: if start_ts is provided, only items with ts <= start_ts are used.

        Notes:
          - `**kwargs` must include `chain_handler` (OptionChainDataHandler) because this runtime handler
            remains a derived layer. Seeding is optional; derived updates can still happen later.
          - Historical source items may be IVSurfaceSnapshot or dicts convertible to IVSurfaceSnapshot.
        """

        # Normalize start_ts -> float|None (avoid heavy deps at import time)
        start_ts_f: float | None
        if start_ts is None:
            start_ts_f = None
        else:
            try:
                import pandas as pd  # local import

                if isinstance(start_ts, pd.Timestamp):
                    start_ts_f = float(start_ts.timestamp())
                else:
                    start_ts_f = float(start_ts)
            except Exception:
                start_ts_f = None

        # Construct runtime handler with the provided config
        rt = cls(symbol=historical_handler.symbol, window=window, **kwargs)

        seed = historical_handler.window(ts=start_ts_f, n=int(window))
        if seed is None:
            rt._anchor_ts = start_ts_f
            return rt

        def _coerce_snapshot(item: Any) -> IVSurfaceSnapshot | None:
            if item is None:
                return None
            if isinstance(item, IVSurfaceSnapshot):
                # enforce symbol if missing
                if item.symbol is None:
                    return IVSurfaceSnapshot.from_surface_aligned(
                        timestamp=float(item.timestamp),
                        data_ts=float(item.timestamp),
                        atm_iv=float(item.atm_iv),
                        skew=float(item.skew),
                        curve=dict(item.curve),
                        surface=dict(item.surface),
                        symbol=rt.symbol,
                        expiry=item.expiry,
                        model=item.model,
                    )
                return item

            if isinstance(item, dict):
                # tolerant dict schema, require keys for from_surface_aligned
                ts_val = item.get("timestamp", item.get("ts"))
                if ts_val is None:
                    return None
                try:
                    data_ts = float(ts_val)
                except Exception:
                    return None

                # Anti-lookahead guard
                if start_ts_f is not None and data_ts > float(start_ts_f):
                    return None

                # Required keys for from_surface_aligned
                atm_iv = item.get("atm_iv")
                skew = item.get("skew")
                curve = item.get("curve")
                surface = item.get("surface")

                if atm_iv is None or skew is None or curve is None or surface is None:
                    return None

                try:
                    atm_iv_f = float(atm_iv)
                    skew_f = float(skew)
                except Exception:
                    return None

                symbol = rt.symbol
                expiry = item.get("expiry")
                expiry_s = expiry if isinstance(expiry, str) and expiry else None

                model = item.get("model")
                model_s = model if isinstance(model, str) and model else rt.model_name

                # Enforce timestamp = engine ts = data_ts (no separate engine ts here, use data_ts)
                timestamp = data_ts

                return IVSurfaceSnapshot.from_surface_aligned(
                    timestamp=timestamp,
                    data_ts=data_ts,
                    atm_iv=atm_iv_f,
                    skew=skew_f,
                    curve=dict(curve) if isinstance(curve, dict) else {},
                    surface=dict(surface) if isinstance(surface, dict) else {},
                    symbol=symbol,
                    expiry=expiry_s,
                    model=model_s,
                )

            return None

        snaps: list[IVSurfaceSnapshot] = []

        # Expect iterable, but tolerate singletons
        try:
            it: Iterable[Any] = seed  # type: ignore[assignment]
            for item in it:
                s = _coerce_snapshot(item)
                if s is not None:
                    snaps.append(s)
        except TypeError:
            s = _coerce_snapshot(seed)
            if s is not None:
                snaps.append(s)

        # Deterministic chronological order
        snaps.sort(key=lambda s: float(s.timestamp))
        for s in snaps:
            rt._snapshots.append(s)

        rt._anchor_ts = start_ts_f

        log_debug(
            rt._logger,
            "IVSurfaceDataHandler.from_historical: seeded snapshots",
            symbol=rt.symbol,
            rows=len(snaps),
            anchor_ts=start_ts_f,
        )

        return rt

    # ------------------------------------------------------------------
    # Lifecycle (realtime/mock)
    # ------------------------------------------------------------------

    def bootstrap(self, *, end_ts: float, lookback: Any | None = None) -> None:
        if lookback is None:
            lookback = self.bootstrap_cfg.get("lookback")
        log_debug(
            self._logger,
            "IVSurfaceDataHandler.bootstrap (no-op)",
            symbol=self.symbol,
            source=self.source,
            end_ts=end_ts,
            lookback=lookback,
        )

    def warmup_to(self, ts: float) -> None:
        self._anchor_ts = float(ts)
        log_debug(self._logger, "IVSurfaceDataHandler warmup_to", symbol=self.symbol, anchor_ts=self._anchor_ts)

    # ------------------------------------------------------------------
    # Derived update (called by engine/driver when appropriate)
    # ------------------------------------------------------------------

    def on_new_tick(self, bar: Any) -> None:
        """Protocol-compatible ingestion.

        Supported payloads:
          - float/int: treated as engine timestamp ts
          - dict with keys: {"timestamp"|"ts"} treated as ts
        """
        ts = _coerce_ts(bar)
        if ts is None:
            return
        snap = self._derive_from_chain(ts)
        if snap is not None:
            self._snapshots.append(snap)

    def _derive_from_chain(self, ts: float) -> IVSurfaceSnapshot | None:
        chain_snap: OptionChainSnapshot | None = self.chain_handler.get_snapshot(ts)
        if chain_snap is None:
            return None

        surface_ts = float(getattr(chain_snap, "timestamp", ts))
        atm_iv = float(getattr(chain_snap, "atm_iv", 0.0))
        skew = float(getattr(chain_snap, "skew", 0.0))
        curve = dict(getattr(chain_snap, "smile", {}))

        # Enforce timestamp = engine ts, data_ts = surface_ts
        return IVSurfaceSnapshot.from_surface_aligned(
            timestamp=ts,
            data_ts=surface_ts,
            atm_iv=atm_iv,
            skew=skew,
            curve=curve,
            surface={},
            symbol=self.symbol,
            expiry=self.expiry,
            model=self.model_name,
        )

    # ------------------------------------------------------------------
    # v4 timestamp-aligned API
    # ------------------------------------------------------------------

    def last_timestamp(self) -> float | None:
        if not self._snapshots:
            return None
        return float(self._snapshots[-1].timestamp)

    def get_snapshot(self, ts: float | None = None) -> Optional[IVSurfaceSnapshot]:
        if ts is None:
            ts = self._anchor_ts if self._anchor_ts is not None else self.last_timestamp()
            if ts is None:
                return None

        t = float(ts)
        for s in reversed(self._snapshots):
            if float(s.timestamp) <= t:
                return s
        return None

    def window(self, ts: float | None = None, n: int = 1) -> list[IVSurfaceSnapshot]:
        if ts is None:
            ts = self._anchor_ts if self._anchor_ts is not None else self.last_timestamp()
            if ts is None:
                return []

        t = float(ts)
        out: list[IVSurfaceSnapshot] = []
        for s in reversed(self._snapshots):
            if float(s.timestamp) <= t:
                out.append(s)
                if len(out) >= int(n):
                    break
        out.reverse()
        return out


def _coerce_ts(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        v = x.get("timestamp", x.get("ts"))
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None
    return None
