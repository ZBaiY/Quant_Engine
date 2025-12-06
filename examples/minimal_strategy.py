from quant_engine.strategy.loader import StrategyLoader
from quant_engine.backtest.engine import BacktestEngine
from quant_engine.data.ohlcv.historical import HistoricalDataHandler

import pandas as pd

# ------------------------------------------------------------------------------
# 1. Example config for the whole strategy
# ------------------------------------------------------------------------------
config = {
    "symbol": "BTCUSDT",

    "features": {
        "RSI": {"period": 14}
    },

    "models": {
        "MAIN": {
            "type": "RSI_MODEL",
            "params": {}
        }
    },

    "decision": {
        "name": "THRESHOLD",
        "params": {"threshold": 0.0}
    },

    "risk": {
        "name": "ATR_SIZER",
        "params": {"window": 14}
    },

    "execution": {
        "policy": {"name": "IMMEDIATE", "params": {}},
        "router": {"name": "SIMPLE", "params": {}},
        "slippage": {"name": "LINEAR", "params": {"b": 0.001}},
        "matching": {"name": "SIMULATED", "params": {}}
    },

    "portfolio": {
        "initial_cash": 10000
    }
}


# ------------------------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------------------------
df = pd.DataFrame({
    "open":  [100, 101, 102, 103],
    "high":  [101, 102, 103, 104],
    "low":   [ 99, 100, 101, 102],
    "close": [100, 101, 102, 103],
    "volume":[10, 20, 15, 18]
})

historical = HistoricalDataHandler.from_dataframe(df)


# ------------------------------------------------------------------------------
# 3. Build strategy with registry + loader
# ------------------------------------------------------------------------------
strategy = StrategyLoader.from_config(config, data_handler=historical)


# ------------------------------------------------------------------------------
# 4. Run backtest with unified execution engine
# ------------------------------------------------------------------------------
engine = BacktestEngine(strategy=strategy, historical=historical)
engine.run()

print("Final portfolio:", engine.portfolio.summary())