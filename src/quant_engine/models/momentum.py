# models/momentum.py
from quant_engine.contracts.model import ModelBase
from .registry import register_model


@register_model("RSI_MODEL")
class RSIMomentumModel(ModelBase):
    required_features = ["RSI"]

    def __init__(self, symbol: str, overbought=70, oversold=30, **kwargs):
        super().__init__(symbol=symbol, **kwargs)
        self.ob = overbought
        self.os = oversold

    def predict(self, features: dict) -> float:
        filtered = self.filter_symbol(features)
        rsi = filtered["RSI_" + self.symbol] if self.symbol else filtered["RSI"]
        if rsi > self.ob:
            return -1.0   # mean reversion short
        elif rsi < self.os:
            return 1.0    # mean reversion long
        return 0.0
    
@register_model("MACD_MODEL")
class MACDMomentumModel(ModelBase):
    required_features = ["MACD", "MACD_SIGNAL"]

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol=symbol, **kwargs)

    def predict(self, features: dict) -> float:
        filtered = self.filter_symbol(features)
        macd = filtered["MACD_" + self.symbol] if self.symbol else filtered["MACD"]
        signal = filtered.get("MACD_SIGNAL_" + self.symbol, 0) if self.symbol else filtered.get("MACD_SIGNAL", 0)
        return 1.0 if macd > signal else -1.0