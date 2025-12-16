from __future__ import annotations
from typing import Any, Optional
import pandas as pd
from quant_engine.data.sentiment.snapshot import SentimentSnapshot
from quant_engine.data.sentiment.cache import SentimentCache
from quant_engine.data.protocol_realtime import RealTimeDataHandler, TimestampLike
from quant_engine.data.sentiment.historical import HistoricalSentimentHandler
from quant_engine.utils.logger import get_logger, log_debug


# TODO : sentiment calculation, pipeline integration, more tests

class SentimentHandler(RealTimeDataHandler):
    """Runtime sentiment handler (mode-agnostic).

    Protocol shadow: subclasses `RealTimeDataHandler`.

    Config (Strategy.DATA.*.sentiment):
      - source: required (e.g. "news", "twitter", "onchain")
      - interval: required cadence (e.g. "15m")
      - model: required (e.g. "embedding", "lexicon", "llm")
      - bootstrap.lookback: convenience horizon for Engine.bootstrap()
      - cache.max_bars: in-memory cache depth (SentimentSnapshot)

    IO boundary:
      - IO-free by default; upstream fetcher lives elsewhere.

    Anti-lookahead:
      - warmup_to(ts) sets anchor; get_snapshot/window clamp to it by default.
    """

    symbol: str
    source: str
    interval: str
    model: str

    bootstrap_cfg: dict[str, Any]
    cache_cfg: dict[str, Any]
    cache: SentimentCache

    _anchor_ts: float | None
    _logger: Any

    def __init__(self, symbol: str, **kwargs: Any):
        self.symbol = symbol
        self._logger = get_logger(__name__)

        src = kwargs.get("source")
        if not isinstance(src, str) or not src:
            raise ValueError("Sentiment handler requires non-empty 'source' (e.g. 'news')")
        self.source = src

        interval = kwargs.get("interval")
        if not isinstance(interval, str) or not interval:
            raise ValueError("Sentiment handler requires non-empty 'interval' (e.g. '15m')")
        self.interval = interval

        model = kwargs.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("Sentiment handler requires non-empty 'model' (e.g. 'embedding')")
        self.model = model

        bootstrap = kwargs.get("bootstrap") or {}
        if not isinstance(bootstrap, dict):
            raise TypeError("Sentiment 'bootstrap' must be a dict")
        self.bootstrap_cfg = dict(bootstrap)

        cache = kwargs.get("cache") or {}
        if not isinstance(cache, dict):
            raise TypeError("Sentiment 'cache' must be a dict")
        self.cache_cfg = dict(cache)

        max_bars = self.cache_cfg.get("max_bars")
        if max_bars is None:
            max_bars = kwargs.get("window", 1000)
        max_bars_i = int(max_bars)
        if max_bars_i <= 0:
            raise ValueError("Sentiment cache.max_bars must be > 0")

        self.cache = SentimentCache(max_bars=max_bars_i)
        self._anchor_ts = None

        log_debug(
            self._logger,
            "SentimentHandler initialized",
            symbol=self.symbol,
            source=self.source,
            interval=self.interval,
            model=self.model,
            max_bars=max_bars_i,
            bootstrap=self.bootstrap_cfg,
        )

    # ------------------------------------------------------------------
    # Lifecycle (realtime/mock)
    # ------------------------------------------------------------------

    def bootstrap(self, *, end_ts: float, lookback: Any | None = None) -> None:
        if lookback is None:
            lookback = self.bootstrap_cfg.get("lookback")
        log_debug(
            self._logger,
            "SentimentHandler.bootstrap (no-op)",
            symbol=self.symbol,
            source=self.source,
            end_ts=end_ts,
            lookback=lookback,
        )

    def warmup_to(self, ts: float) -> None:
        self._anchor_ts = float(ts)
        log_debug(self._logger, "SentimentHandler warmup_to", symbol=self.symbol, anchor_ts=self._anchor_ts)

    # ------------------------------------------------------------------
    # Streaming ingestion
    # ------------------------------------------------------------------

    def on_new_tick(self, bar: Any) -> None:
        """Accept SentimentSnapshot or dict payloads."""
        snap = self._coerce_snapshot(bar, engine_ts=self.last_timestamp() or 0.0)
        if snap is None:
            return
        self.cache.update(snap)

    # ------------------------------------------------------------------
    # Timestamp-aligned access
    # ------------------------------------------------------------------

    def last_timestamp(self) -> float | None:
        s = self.cache.latest()
        return None if s is None else float(s.timestamp)

    def get_snapshot(self, ts: float | None = None) -> Optional[SentimentSnapshot]:
        if ts is None:
            ts = self._anchor_ts if self._anchor_ts is not None else self.last_timestamp()
            if ts is None:
                return None
        return self.cache.latest_before_ts(float(ts))

    def window(self, ts: float | None = None, n: int = 1) -> list[SentimentSnapshot]:
        if ts is None:
            ts = self._anchor_ts if self._anchor_ts is not None else self.last_timestamp()
            if ts is None:
                return []
        return self.cache.window_before_ts(float(ts), int(n))

    def reset(self) -> None:
        self.cache.clear()
    
    @classmethod
    def from_historical(
        cls,
        historical_handler: HistoricalSentimentHandler,
        *,
        start_ts: float | pd.Timestamp | None = None,
        window: int = 1000,
        **kwargs: Any,
    ) -> "SentimentHandler":
        init_kwargs = dict(kwargs)
        init_kwargs.setdefault("window", window)
        rt = cls(symbol=historical_handler.symbol, **init_kwargs)

        start_ts_f = _to_float_ts(start_ts)
        seed = historical_handler.window(ts=start_ts_f, n=int(window))
        if not seed:
            rt._anchor_ts = start_ts_f
            return rt

        engine_ts = start_ts_f
        if engine_ts is None:
            last = historical_handler.last_timestamp()
            engine_ts = float(last) if last is not None else 0.0

        try:
            for item in seed:
                snap = rt._coerce_snapshot(item, engine_ts=engine_ts)
                if snap is not None:
                    rt.cache.update(snap)
        except TypeError:
            snap = rt._coerce_snapshot(seed, engine_ts=engine_ts)
            if snap is not None:
                rt.cache.update(snap)

        rt._anchor_ts = start_ts_f
        return rt

    def _coerce_snapshot(self, item: Any, *, engine_ts: float) -> SentimentSnapshot | None:
        if isinstance(item, SentimentSnapshot):
            return item

        if not isinstance(item, dict):
            return None

        obs_ts = item.get("obs_ts", item.get("timestamp", item.get("ts")))
        if obs_ts is None:
            return None
        try:
            obs_ts_f = float(obs_ts)
        except Exception:
            return None

        score = item.get("score", 0.0)
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0

        emb = item.get("embedding")
        if emb is not None and not isinstance(emb, list):
            emb = None
        embedding = [float(x) for x in emb] if isinstance(emb, list) else None

        meta = item.get("meta")
        meta_d = meta if isinstance(meta, dict) else {}

        # IMPORTANT: use handler's canonical metadata (cfg-driven), not item['source']/item['model']
        return SentimentSnapshot.from_payload(
            engine_ts=engine_ts,
            obs_ts=obs_ts_f,
            symbol=self.symbol,
            source=self.source,
            interval=self.interval,
            model=self.model,
            score=score_f,
            embedding=embedding,
            meta=meta_d,
        )
        

def _to_float_ts(ts: "TimestampLike | None") -> float | None:
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        return float(ts.timestamp())
    return float(ts)

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

