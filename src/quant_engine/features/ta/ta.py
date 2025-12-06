# features/ta.py
from quant_engine.contracts.feature import FeatureChannel
from ..registry import register_feature
import pandas as pd
from quant_engine.utils.logger import get_logger, log_debug

@register_feature("RSI")
class RSIFeature(FeatureChannel):
    _logger = get_logger(__name__)
    def __init__(self, symbol=None, **kwargs):
        self.symbol = symbol
        self.period = kwargs.get("period", 14)

    def compute(self, data: pd.DataFrame):
        """
        sub = df[df.symbol == self.symbol]
        ...
        return {f"RSI_{self.symbol}": rsi_value}
        --- for multiple symbols, filter data first
        """
        log_debug(self._logger, "RSIFeature compute() called")
        delta = data["close"].diff()
        up = delta.clip(lower=0).rolling(self.period).mean()
        down = (-delta.clip(upper=0)).rolling(self.period).mean()
        rsi = up.iloc[-1] / (up.iloc[-1] + down.iloc[-1] + 1e-12)
        result = {"rsi": float(rsi)}
        log_debug(self._logger, "RSIFeature output", value=result["rsi"])
        return result


@register_feature("MACD")
class MACDFeature(FeatureChannel):
    _logger = get_logger(__name__)
    def __init__(self, symbol=None, **kwargs):

        self.symbol = symbol
        self.fast = kwargs.get("fast", 12)
        self.slow = kwargs.get("slow", 26)
        self.signal = kwargs.get("signal", 9)

    def compute(self, data):
        log_debug(self._logger, "MACDFeature compute() called")
        fast_ema = data["close"].ewm(span=self.fast).mean()
        slow_ema = data["close"].ewm(span=self.slow).mean()
        macd = fast_ema - slow_ema
        result = {"macd": float(macd.iloc[-1])}
        log_debug(self._logger, "MACDFeature output", value=result["macd"])
        return result