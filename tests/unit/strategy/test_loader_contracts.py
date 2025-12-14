import pytest
from quant_engine.strategy.loader import StrategyLoader
from quant_engine.strategy.engine import StrategyEngine
from quant_engine.features.extractor import FeatureExtractor

from quant_engine.data.ohlcv.realtime import RealTimeDataHandler
from quant_engine.data.orderbook.realtime import RealTimeOrderbookHandler
from quant_engine.data.derivatives.option_chain.chain_handler import OptionChainDataHandler
from quant_engine.data.derivatives.iv.iv_handler import IVSurfaceDataHandler
from quant_engine.data.sentiment.loader import SentimentLoader


# ---------------------------------------------------------------------
# Test config (verbatim from user)
# ---------------------------------------------------------------------
@pytest.fixture
def strategy_config():
    return {
        "symbol": "BTCUSDT",

        "features_user": [
            { "type": "SPREAD", "symbol": "BTCUSDT", "params": {"ref": "ETHUSDT"} },
            { "type": "RSI",    "symbol": "BTCUSDT" },
            { "type": "RSI",    "symbol": "ETHUSDT" }
        ],

        "model": {
            "type": "PAIR_ZSCORE",
            "params": {
                "secondary": "ETHUSDT",
                "lookback": 120
            }
        },
        "decision": {
            "type": "ZSCORE_THRESHOLD",
            "params": {
                "enter": 2.0,
                "exit": 0.5
            }
        },

        "risk": {
            "rules": {
                "ATR_SIZER": {},
                "EXPOSURE_LIMIT": { "params": { "limit": 2.0 } }
            }
        },

        "execution": {
            "policy":   { "type": "IMMEDIATE" },
            "router":   { "type": "SIMPLE" },
            "slippage": { "type": "LINEAR" },
            "matching": { "type": "SIMULATED" }
        },

        "portfolio": {
            "type": "STANDARD",
            "params": { "initial_capital": 10000 }
        }
    }


# ---------------------------------------------------------------------
# Main pipeline test
# ---------------------------------------------------------------------
def test_strategy_loader_full_pipeline(strategy_config):
    engine = StrategyLoader.from_config(strategy_config)

    # ---- basic assembly ----
    assert isinstance(engine, StrategyEngine)
    assert engine.symbol == "BTCUSDT"

    # ---- handlers wiring ----
    assert isinstance(engine.ohlcv_handlers, dict)
    assert set(engine.ohlcv_handlers.keys()) == {"BTCUSDT", "ETHUSDT"}
    for h in engine.ohlcv_handlers.values():
        assert isinstance(h, RealTimeDataHandler)

    # optional layers may or may not exist, but types must be correct
    for h in engine.orderbook_handlers.values():
        assert isinstance(h, RealTimeOrderbookHandler)

    for h in engine.option_chain_handlers.values():
        assert isinstance(h, OptionChainDataHandler)

    for h in engine.iv_surface_handlers.values():
        assert isinstance(h, IVSurfaceDataHandler)

    for h in engine.sentiment_handlers.values():
        assert isinstance(h, SentimentLoader)

    # ---- feature extractor ----
    fe = engine.feature_extractor
    assert isinstance(fe, FeatureExtractor)

    # resolved features must include:
    # - core features (ATR, VOLATILITY) for BTCUSDT
    # - user RSI for BTCUSDT
    # - user RSI for ETHUSDT
    # - SPREAD(BTCUSDT, ref=ETHUSDT)
    feature_cfg = fe.feature_config

    def fkey(f):
        return (f["type"], f.get("symbol"), tuple(sorted(f.get("params", {}).items())))

    keys = {fkey(f) for f in feature_cfg}

    assert ("ATR", "BTCUSDT", ()) in keys
    assert ("RSI", "BTCUSDT", ()) in keys
    assert ("RSI", "ETHUSDT", ()) in keys
    assert ("SPREAD", "BTCUSDT", (("ref", "ETHUSDT"),)) in keys

    # ---- model ----
    assert "main" in engine.models
    model = engine.models["main"]
    assert model.symbol == "BTCUSDT"

    # model secondary symbol must be preserved
    assert getattr(model, "secondary", None) == "ETHUSDT"

    # ---- decision / risk / execution / portfolio ----
    assert engine.decision is not None
    assert engine.risk_manager is not None
    assert engine.execution_engine is not None
    assert engine.portfolio is not None

    # ---- no accidental symbol leakage ----
    assert engine.portfolio.symbol == "BTCUSDT"
    assert engine.execution_engine is not None