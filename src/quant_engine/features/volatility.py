# src/quant_engine/features/volatility.py
import pandas as pd
from quant_engine.contracts.feature import FeatureChannel
from .registry import register_feature
from quant_engine.utils.logger import get_logger, log_debug


@register_feature("ATR")
class ATRFeature(FeatureChannel):
    _logger = get_logger(__name__)
    """Average True Range."""
    def __init__(self, period=14):
        self.period = period

    def compute(self, df: pd.DataFrame):
        log_debug(self._logger, "ATRFeature compute() called")
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(self.period).mean().iloc[-1]
        result = {"atr": float(atr)}
        log_debug(self._logger, "ATRFeature output", value=result["atr"])
        return result


@register_feature("REALIZED_VOL")
class RealizedVolFeature(FeatureChannel):
    _logger = get_logger(__name__)
    """Realized volatility via daily returns."""
    def __init__(self, window=30):
        self.window = window

    def compute(self, df: pd.DataFrame):
        log_debug(self._logger, "RealizedVolFeature compute() called")
        returns = df["close"].pct_change().dropna()
        vol = returns.rolling(self.window).std().iloc[-1]
        result = {"realized_vol": float(vol)}
        log_debug(self._logger, "RealizedVolFeature output", value=result["realized_vol"])
        return result