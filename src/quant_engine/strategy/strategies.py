"""
Concrete strategy definitions.

This module contains registered Strategy implementations that declare
their data and feature dependencies.
"""

from quant_engine.strategy.base import StrategyBase
from quant_engine.strategy.registry import register_strategy


@register_strategy("EXAMPLE")
class ExampleStrategy(StrategyBase):
    """
    Pair trading via Z-score.

    Python-first declarative strategy specification.
    This class is the single source of truth for the experiment setup.
    """

    STRATEGY_NAME = "EXAMPLE"

    # --------------------------------------------------
    # Primary symbol
    # --------------------------------------------------
    SYMBOL = "BTCUSDT"

    # --------------------------------------------------
    # Data declaration (resource layer)
    # --------------------------------------------------
    DATA = {
        "primary": {
            "ohlcv": {
                "source": "binance",
                "interval": "1m",
                "history": {
                    "lookback": "30d",
                    "warmup": 200,
                },
            },
            "option_chain": {
                "source": "deribit",
                "refresh_interval": "5m",
            },
            "orderbook": {
                "depth": 10,
                "aggregation": "L2",
                "refresh_interval": "100ms",
            },
        },
        "secondary": {
            "ETHUSDT": {
                "ohlcv": {
                    "source": "binance",
                    "interval": "1m",
                    "history": {
                        "lookback": "30d",
                        "warmup": 200,
                    },
                }
            }
        },
    }

    # Data domains required by this strategy type
    REQUIRED_DATA = {
        "ohlcv",
        "orderbook",
        "option_chain",
    }

    # --------------------------------------------------
    # Feature layer
    # --------------------------------------------------
    FEATURES_USER = [
        {
            "name": "SPREAD_MODEL_BTCUSDT^ETHUSDT",
            "type": "SPREAD",
            "symbol": "BTCUSDT",
            "params": {"ref": "ETHUSDT"},
        },
        {
            "name": "RSI_MODEL_BTCUSDT",
            "type": "RSI",
            "symbol": "BTCUSDT",
            "params": {"window": 14},
        },
        {
            "name": "RSI_MODEL_ETHUSDT",
            "type": "RSI",
            "symbol": "ETHUSDT",
            "params": {"window": 14},
        },
        {
            "name": "ZSCORE_MODEL_ETHUSDT^BTCUSDT",
            "type": "ZSCORE",
            "symbol": "BTCUSDT",
            "params": {
                "secondary": "ETHUSDT",
                "lookback": 120,
            },
        },
    ]

    # --------------------------------------------------
    # Model / Decision / Risk
    # --------------------------------------------------
    MODEL_CFG = {
        "type": "PAIR_ZSCORE",
        "params": {
            "zscore_feature": "ZSCORE_MODEL_ETHUSDT^BTCUSDT",
        },
    }

    DECISION_CFG = {
        "type": "ZSCORE_THRESHOLD",
        "params": {
            "zscore_feature": "ZSCORE_MODEL_ETHUSDT^BTCUSDT",
            "enter": 2.0,
            "exit": 0.5,
        },
    }

    RISK_CFG = {
        "rules": {
            "ATR_SIZER": {
                "params": {
                    "atr_feature": "ATR_RISK_BTCUSDT",
                }
            },
            "EXPOSURE_LIMIT": {
                "params": {"limit": 2.0},
            },
        }
    }

    # --------------------------------------------------
    # Execution / Portfolio
    # --------------------------------------------------
    EXECUTION_CFG = {
        "policy": {"type": "IMMEDIATE"},
        "router": {"type": "SIMPLE"},
        "slippage": {"type": "LINEAR"},
        "matching": {"type": "SIMULATED"},
    }

    PORTFOLIO_CFG = {
        "type": "STANDARD",
        "params": {
            "initial_capital": 10000,
        },
    }


