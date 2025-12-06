from quant_engine.contracts.risk import RiskProto
from quant_engine.risk.registry import register_risk
from quant_engine.utils.logger import get_logger, log_debug

@register_risk("SENTIMENT_SCALE")
class SentimentScaleRule(RiskProto):
    _logger = get_logger(__name__)
    def __init__(self, key="sentiment_score", strength=1.0):
        self.key = key
        self.strength = strength
        log_debug(self._logger, "SentimentScaleRule initialized", key=key, strength=strength)

    def adjust(self, size: float, features: dict) -> float:
        log_debug(self._logger, "SentimentScaleRule adjust() called", size=size, sentiment=features.get(self.key))
        sentiment = features.get(self.key, 0.0)
        result = size * (1 + self.strength * sentiment)
        log_debug(self._logger, "SentimentScaleRule output", adjusted_size=result)
        return result