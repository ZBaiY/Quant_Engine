import pytest
from typing import Dict, Any

from quant_engine.features.loader import FeatureLoader
from quant_engine.features.extractor import FeatureExtractor


class DummyHandler:
    """Minimal stand-in for any data handler type."""
    def __init__(self, name: str) -> None:
        self.name = name


# ---------------------------------------------------------------------------
# Basic wiring: empty feature config, multiple handler types
# ---------------------------------------------------------------------------


def test_from_config_preserves_handler_mappings_with_empty_config():
    """
    When feature_config_list is empty, FeatureLoader should still
    construct a FeatureExtractor and pass all handler mappings through
    by identity.
    """
    ohlcv_handlers: Dict[str, Any] = {
        "BTCUSDT": DummyHandler("ohlcv_btc"),
        "ETHUSDT": DummyHandler("ohlcv_eth"),
    }
    orderbook_handlers: Dict[str, Any] = {
        "BTCUSDT": DummyHandler("ob_btc"),
    }
    option_chain_handlers: Dict[str, Any] = {
        "BTCUSDT": DummyHandler("opt_btc"),
    }
    iv_surface_handlers: Dict[str, Any] = {
        "BTCUSDT": DummyHandler("iv_btc"),
    }
    sentiment_handlers: Dict[str, Any] = {
        "BTCUSDT": DummyHandler("sent_btc"),
    }

    fe = FeatureLoader.from_config(
        feature_config_list=[],
        ohlcv_handlers=ohlcv_handlers,
        orderbook_handlers=orderbook_handlers,
        option_chain_handlers=option_chain_handlers,
        iv_surface_handlers=iv_surface_handlers,
        sentiment_handlers=sentiment_handlers,
    )

    assert isinstance(fe, FeatureExtractor)

    # Identity, not copies
    assert fe.ohlcv_handlers is ohlcv_handlers
    assert fe.orderbook_handlers is orderbook_handlers
    assert fe.option_chain_handlers is option_chain_handlers
    assert fe.iv_surface_handlers is iv_surface_handlers
    assert fe.sentiment_handlers is sentiment_handlers


# ---------------------------------------------------------------------------
# Contract: full multi-datatype wiring + feature config
# ---------------------------------------------------------------------------


def test_from_config_passes_all_handlers_and_feature_config(monkeypatch):
    """
    FeatureLoader should pass all five handler mappings and the feature
    config list into FeatureExtractor's constructor unchanged.

    We monkeypatch FeatureExtractor in the loader module to a dummy
    implementation to avoid depending on registry/build_feature.
    """
    import quant_engine.features.loader as loader_mod

    captured: Dict[str, Any] = {}

    class DummyExtractor:
        def __init__(
            self,
            *,
            ohlcv_handlers,
            orderbook_handlers,
            option_chain_handlers,
            iv_surface_handlers,
            sentiment_handlers,
            feature_config,
        ) -> None:
            captured["ohlcv_handlers"] = ohlcv_handlers
            captured["orderbook_handlers"] = orderbook_handlers
            captured["option_chain_handlers"] = option_chain_handlers
            captured["iv_surface_handlers"] = iv_surface_handlers
            captured["sentiment_handlers"] = sentiment_handlers
            captured["feature_config"] = feature_config

    monkeypatch.setattr(loader_mod, "FeatureExtractor", DummyExtractor)

    # Multi-symbol, multi-datatype handler dicts
    ohlcv_handlers = {
        "BTCUSDT": DummyHandler("ohlcv_btc"),
        "ETHUSDT": DummyHandler("ohlcv_eth"),
    }
    orderbook_handlers = {
        "BTCUSDT": DummyHandler("ob_btc"),
        "ETHUSDT": DummyHandler("ob_eth"),
    }
    option_chain_handlers = {
        "BTCUSDT": DummyHandler("opt_btc"),
    }
    iv_surface_handlers = {
        "BTCUSDT": DummyHandler("iv_btc"),
    }
    sentiment_handlers = {
        "BTCUSDT": DummyHandler("sent_btc"),
    }

    feature_config = [
        {"type": "RSI", "symbol": "BTCUSDT", "params": {"period": 14}},
        {"type": "SPREAD", "symbol": "ETHUSDT", "params": {"ref": "BTCUSDT"}},
    ]

    extractor = FeatureLoader.from_config(
        feature_config_list=feature_config,
        ohlcv_handlers=ohlcv_handlers,
        orderbook_handlers=orderbook_handlers,
        option_chain_handlers=option_chain_handlers,
        iv_surface_handlers=iv_surface_handlers,
        sentiment_handlers=sentiment_handlers,
    )

    # We actually got our DummyExtractor
    assert isinstance(extractor, DummyExtractor)

    # All handler mappings passed through by identity
    assert captured["ohlcv_handlers"] is ohlcv_handlers
    assert captured["orderbook_handlers"] is orderbook_handlers
    assert captured["option_chain_handlers"] is option_chain_handlers
    assert captured["iv_surface_handlers"] is iv_surface_handlers
    assert captured["sentiment_handlers"] is sentiment_handlers

    # Feature config passed through unchanged
    assert captured["feature_config"] == feature_config
    assert captured["feature_config"][0]["type"] == "RSI"
    assert captured["feature_config"][1]["params"]["ref"] == "BTCUSDT"