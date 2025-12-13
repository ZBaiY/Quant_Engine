import pytest
from typing import Any, Dict, List, cast

import pandas as pd

from quant_engine.features.extractor import FeatureExtractor, min_warmup
from quant_engine.data.ohlcv.realtime import RealTimeDataHandler
from quant_engine.data.orderbook.realtime import RealTimeOrderbookHandler
from quant_engine.data.derivatives.option_chain.chain_handler import OptionChainDataHandler
from quant_engine.data.derivatives.iv.iv_handler import IVSurfaceDataHandler
from quant_engine.data.sentiment.loader import SentimentLoader
from quant_engine.contracts.feature import FeatureChannel
import quant_engine.features as _pkg  # anchor import

# ---------------------------------------------------------------------------
# Dummy FeatureChannels for contract testing
# ---------------------------------------------------------------------------


class DummyChannel:
    """
    Minimal feature channel for exercising FeatureExtractor contracts.

    Implements:
        - required_window()
        - initialize(context, warmup_window)
        - update(context)
        - output()
        - symbol
        - params
    """

    def __init__(self, name: str, symbol: str | None = None, **params: Any) -> None:
        self.name = name
        self.symbol = symbol
        self.params: Dict[str, Any] = params or {}

        self.required_window_value: int = 1
        self.initialize_calls: List[Dict[str, Any]] = []
        self.update_calls: List[Dict[str, Any]] = []
        self.output_value: Dict[str, Any] = {name: 1.0}

    def required_window(self) -> int:
        return self.required_window_value

    def initialize(self, context: Dict[str, Any], warmup_window: int) -> None:
        self.initialize_calls.append(
            {"context": context, "warmup_window": warmup_window}
        )

    def update(self, context: Dict[str, Any]) -> None:
        self.update_calls.append({"context": context})

    def output(self) -> Dict[str, Any]:
        return self.output_value


# ---------------------------------------------------------------------------
# Dummy handlers for OHLCV and others
# ---------------------------------------------------------------------------



class DummyOHLCVHandler(RealTimeDataHandler):
    """
    Test-only subclass of RealTimeDataHandler exposing just the pieces
    needed by FeatureExtractor.initialize()/update().

    IMPORTANT:
    - We **do** call the real __init__ so that all base-class behaviour
      (including DataCache, window_df, last_timestamp) is exercised.
    - We only add lightweight instrumentation (window_df_calls) on top.
    """

    def __init__(self, last_ts: float = 100.0) -> None:
        # Use real init to set up cache & logger
        super().__init__(symbol="BTCUSDT", window=1000)
        self.window_df_calls: List[int] = []

        # Seed the handler with a single bar so last_timestamp() is defined.
        df = pd.DataFrame(
            [
                {
                    "timestamp": last_ts,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1.0,
                }
            ]
        )
        # Use the real ingestion path
        self.on_new_tick(df)

    def window_df(self, n: int):
        """
        Track calls while delegating to the real implementation.
        """
        self.window_df_calls.append(n)
        return super().window_df(n)


# ---------------------------------------------------------------------------
# Helper to build a FeatureExtractor with minimal handlers
# ---------------------------------------------------------------------------


def make_extractor_with_channels(channels: List[DummyChannel]) -> FeatureExtractor:
    # OHLCV: use the test-specific subclass so we still track window_df calls.
    ohlcv_handlers: Dict[str, RealTimeDataHandler] = {
        "BTCUSDT": DummyOHLCVHandler(last_ts=123.0)
    }

    # Orderbook: real realtime handler, no data needed for these contract tests.
    orderbook_handlers: Dict[str, RealTimeOrderbookHandler] = {
        "BTCUSDT": RealTimeOrderbookHandler(symbol="BTCUSDT")
    }

    # Options: real chain handler; left empty because FeatureExtractor
    # contract tests never touch its internals.
    option_chain_handlers: Dict[str, OptionChainDataHandler] = {
        "BTCUSDT": OptionChainDataHandler(symbol="BTCUSDT")
    }

    # IV surface: real handler derived from the option chain handler.
    iv_surface_handlers: Dict[str, IVSurfaceDataHandler] = {
        "BTCUSDT": IVSurfaceDataHandler(
            symbol="BTCUSDT",
            chain_handler=option_chain_handlers["BTCUSDT"],
        )
    }

    # Sentiment: use the real loader; we don't call it in these tests,
    # only pass it through the FeatureExtractor context.
    sentiment_handlers: Dict[str, SentimentLoader] = {
        "BTCUSDT": SentimentLoader(symbol="BTCUSDT")
    }

    fe = FeatureExtractor(
        ohlcv_handlers=cast(Dict[str, RealTimeDataHandler], ohlcv_handlers),
        orderbook_handlers=orderbook_handlers,
        option_chain_handlers=option_chain_handlers,
        iv_surface_handlers=iv_surface_handlers,
        sentiment_handlers=sentiment_handlers,
        feature_config=[],
    )
    # Bypass build_feature: inject our dummy channels directly.
    fe.channels = cast(List[FeatureChannel], channels)
    return fe


# ---------------------------------------------------------------------------
# __init__ contract: build_feature is called correctly
# ---------------------------------------------------------------------------


def test_init_uses_build_feature_for_each_config(monkeypatch):

    # Import module to patch build_feature symbol in the extractor module
    import quant_engine.features.extractor as extractor_mod

    calls: List[Dict[str, Any]] = []

    def fake_build_feature(type_name: str, symbol: str | None = None, **params: Any):
        calls.append({"type": type_name, "symbol": symbol, "params": params})
        # return DummyChannel so extractor.channels has real objects
        return DummyChannel(name=type_name, symbol=symbol, **params)

    monkeypatch.setattr(extractor_mod, "build_feature", fake_build_feature)

    cfg = [
        {"type": "RSI", "symbol": "BTCUSDT", "params": {"period": 14}},
        {"type": "SPREAD", "symbol": "ETHUSDT", "params": {"ref": "BTCUSDT"}},
    ]

    fe = FeatureExtractor(
        ohlcv_handlers=cast(Dict[str, RealTimeDataHandler], {"BTCUSDT": DummyOHLCVHandler()}),
        orderbook_handlers={},
        option_chain_handlers={},
        iv_surface_handlers={},
        sentiment_handlers={},
        feature_config=cfg,
    )

    assert len(calls) == 2
    assert len(fe.channels) == 2

    assert calls[0] == {
        "type": "RSI",
        "symbol": "BTCUSDT",
        "params": {"period": 14},
    }
    assert calls[1] == {
        "type": "SPREAD",
        "symbol": "ETHUSDT",
        "params": {"ref": "BTCUSDT"},
    }

    assert isinstance(fe.channels[0], DummyChannel)
    assert fe.channels[0].symbol == "BTCUSDT"
    assert fe.channels[0].params["period"] == 14


def test_init_with_none_feature_config_creates_no_channels(monkeypatch):
    import quant_engine.features.extractor as extractor_mod

    def fake_build_feature(*args: Any, **kwargs: Any):
        raise AssertionError("build_feature should not be called when feature_config is None")

    monkeypatch.setattr(extractor_mod, "build_feature", fake_build_feature)

    fe = FeatureExtractor(
        ohlcv_handlers=cast(Dict[str, RealTimeDataHandler], {"BTCUSDT": DummyOHLCVHandler()}),
        orderbook_handlers={},
        option_chain_handlers={},
        iv_surface_handlers={},
        sentiment_handlers={},
        feature_config=None,
    )

    assert fe.channels == []


# ---------------------------------------------------------------------------
# initialize() contract
# ---------------------------------------------------------------------------


def test_initialize_uses_max_required_window_and_min_warmup():
    ch_small = DummyChannel("small", symbol="BTCUSDT")
    ch_big = DummyChannel("big", symbol="BTCUSDT")
    ch_small.required_window_value = 50
    ch_big.required_window_value = 500

    fe = make_extractor_with_channels([ch_small, ch_big])

    out = fe.initialize()

    # warmup_window is max(max_window, min_warmup)
    # here max_window = 500, min_warmup = min_warmup => warmup_window = 500
    assert len(ch_small.initialize_calls) == 1
    assert len(ch_big.initialize_calls) == 1
    assert ch_small.initialize_calls[0]["warmup_window"] == 500
    assert ch_big.initialize_calls[0]["warmup_window"] == 500

    # internal state consistent with primary handler
    handler = next(iter(fe.ohlcv_handlers.values()))
    assert fe._initialized is True
    assert fe._last_ts == handler.last_timestamp()
    assert fe._last_output == out


def test_initialize_with_no_channels_uses_min_warmup_only():
    fe = make_extractor_with_channels([])

    out = fe.initialize()

    # With no channels, default max_window = 1 → warmup = max(1, min_warmup) = min_warmup
    # There is no channel to observe warmup_window directly; we only verify
    # that initialization completes and internal state is set.
    handler = next(iter(fe.ohlcv_handlers.values()))
    assert fe._initialized is True
    assert fe._last_ts == handler.last_timestamp()
    assert fe._last_output == out

# ---------------------------------------------------------------------------
# update() contract
# ---------------------------------------------------------------------------


def test_update_calls_initialize_when_not_initialized():
    ch = DummyChannel("foo", symbol="BTCUSDT")
    fe = make_extractor_with_channels([ch])

    assert fe._initialized is False
    out = fe.update(ts=999.0)

    # initialize should have been called once
    assert len(ch.initialize_calls) == 1
    # update should not be called during first warmup
    assert len(ch.update_calls) == 0
    assert fe._initialized is True
    assert fe._last_output == out


def test_update_infers_ts_from_primary_handler_when_none():
    ch = DummyChannel("foo", symbol="BTCUSDT")
    fe = make_extractor_with_channels([ch])

    # manually mark as initialized and set _last_ts older than handler
    fe._initialized = True
    fe._last_ts = 10.0

    handler = next(iter(fe.ohlcv_handlers.values()))
    assert handler.last_timestamp() == 123.0
    
    out = fe.update(ts=None)

    assert len(ch.update_calls) == 1
    ctx = ch.update_calls[0]["context"]
    assert ctx["ts"] == handler.last_timestamp()
    assert fe._last_ts == handler.last_timestamp()
    assert fe._last_output == out


def test_update_uses_fallback_ts_when_no_handlers_and_last_ts_exists():
    ch = DummyChannel("foo", symbol=None)
    fe = FeatureExtractor(
        ohlcv_handlers={},  # no primary handler
        orderbook_handlers={},
        option_chain_handlers={},
        iv_surface_handlers={},
        sentiment_handlers={},
        feature_config=[],
    )
    fe.channels = cast(List[FeatureChannel], [ch])
    fe._initialized = True
    fe._last_ts = 42.0
    fe._last_output = {"PREV": 1.0}

    out = fe.update(ts=None)

    # ts fallback 到 _last_ts，但因为“没有前进”，使用缓存输出，不调用 update
    assert len(ch.update_calls) == 0
    assert out == {"PREV": 1.0}
    assert fe._last_ts == 42.0


def test_update_uses_zero_ts_when_no_handlers_and_no_last_ts():
    ch = DummyChannel("foo", symbol=None)
    fe = FeatureExtractor(
        ohlcv_handlers={},
        orderbook_handlers={},
        option_chain_handlers={},
        iv_surface_handlers={},
        sentiment_handlers={},
        feature_config=[],
    )
    fe.channels = cast(List[FeatureChannel], [ch])
    fe._initialized = True
    fe._last_ts = None

    out = fe.update(ts=None)

    assert len(ch.update_calls) == 1
    ctx = ch.update_calls[0]["context"]
    assert ctx["ts"] == 0.0
    assert fe._last_ts == 0.0
    assert fe._last_output == out


def test_update_returns_cached_output_when_ts_not_advanced():
    ch = DummyChannel("foo", symbol="BTCUSDT")
    fe = make_extractor_with_channels([ch])

    fe._initialized = True
    fe._last_ts = 100.0
    fe._last_output = {"PREV": 1.0}

    out = fe.update(ts=90.0)

    # no update() call on channels
    assert len(ch.update_calls) == 0
    assert out == {"PREV": 1.0}
    assert fe._last_output == {"PREV": 1.0}
    assert fe._last_ts == 100.0


# ---------------------------------------------------------------------------
# compute_output() contract
# ---------------------------------------------------------------------------


def test_compute_output_key_format_with_symbol_and_ref():
    ch1 = DummyChannel("rsi", symbol="BTCUSDT")
    ch1.output_value = {"rsi": 50.0}
    ch1.params = {}  # no ref

    ch2 = DummyChannel("spread", symbol="BTCUSDT")
    ch2.output_value = {"spread": 0.1}
    ch2.params = {"ref": "ETHUSDT"}

    ch3 = DummyChannel("raw", symbol=None)
    ch3.output_value = {"raw": 1.23}
    ch3.params = {}

    fe = make_extractor_with_channels([ch1, ch2, ch3])

    out = fe.compute_output()

    assert out["RSI_BTCUSDT"] == 50.0
    assert out["SPREAD_ETHUSDT^BTCUSDT"] == 0.1
    assert out["RAW"] == 1.23


def test_compute_calls_update_and_returns_its_result(monkeypatch):
    fe = make_extractor_with_channels([])

    calls: List[Dict[str, Any]] = []

    def fake_update(ts: float | None = None) -> Dict[str, Any]:
        calls.append({"ts": ts})
        return {"X": 1.0}

    monkeypatch.setattr(fe, "update", fake_update)

    out = fe.compute()

    assert calls == [{"ts": None}]
    assert out == {"X": 1.0}

def test_update_infers_ts_from_all_handlers_when_ohlcv_missing(monkeypatch):
    """
    When ts=None and there is no OHLCV handler, FeatureExtractor.update()
    must infer ts from *all* handler families that expose last_timestamp(),
    and use the maximum of those timestamps.
    """
    ch = DummyChannel("foo", symbol="BTCUSDT")

    # Real handlers, but we control their last_timestamp via monkeypatch.
    ob_handler = RealTimeOrderbookHandler(symbol="BTCUSDT")
    iv_handler = IVSurfaceDataHandler(
        symbol="BTCUSDT",
        chain_handler=OptionChainDataHandler(symbol="BTCUSDT"),
    )
    sent_handler = SentimentLoader(symbol="BTCUSDT")

    # Orderbook last_ts = 150
    monkeypatch.setattr(ob_handler, "last_timestamp", lambda: 150.0)
    # IV surface last_ts = 200
    monkeypatch.setattr(iv_handler, "last_timestamp", lambda: 200.0)
    # Sentiment last_ts = 50
    monkeypatch.setattr(sent_handler, "last_timestamp", lambda: 50.0)

    fe = FeatureExtractor(
        ohlcv_handlers={},  # **no primary OHLCV handler**
        orderbook_handlers={"BTCUSDT": ob_handler},
        option_chain_handlers={},  # not used for ts inference
        iv_surface_handlers={"BTCUSDT": iv_handler},
        sentiment_handlers={"BTCUSDT": sent_handler},
        feature_config=[],
    )
    fe.channels = cast(List[FeatureChannel], [ch])
    fe._initialized = True
    fe._last_ts = None

    out = fe.update(ts=None)

    # One incremental update call on our dummy channel
    assert len(ch.update_calls) == 1
    ctx = ch.update_calls[0]["context"]

    # ts must be the max over all handler timestamps → 200.0 (from iv_handler)
    assert ctx["ts"] == pytest.approx(200.0)
    assert fe._last_ts == pytest.approx(200.0)
    assert fe._last_output == out