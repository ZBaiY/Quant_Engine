from typing import Dict, Any
from quant_engine.contracts.risk import RiskBase
from quant_engine.risk.registry import register_risk
from quant_engine.utils.logger import get_logger, log_debug


@register_risk("SENTIMENT_SCALE")
class SentimentScaleRule(RiskBase):
    """
    V4 sentiment-based risk scaler.

    Contracts:
    - symbol-aware via RiskBase
    - optional sentiment feature dependency
    - adjust(size, features) -> float
    """

    required_features: list[str] = []

    _logger = get_logger(__name__)

    def __init__(
        self,
        symbol: str,
        key: str = "sentiment_score",
        strength: float = 1.0,
        **kwargs,
    ):
        super().__init__(symbol=symbol)
        self.key = key
        self.strength = strength
        log_debug(
            self._logger,
            "SentimentScaleRule initialized",
            symbol=symbol,
            key=key,
            strength=strength,
        )

    def adjust(self, size: float, features: Dict[str, Any]) -> float:
        sentiment = features.get(self.key, 0.0)

        log_debug(
            self._logger,
            "SentimentScaleRule adjust() called",
            size=size,
            sentiment=sentiment,
        )

        adjusted = size * (1.0 + self.strength * sentiment)

        log_debug(
            self._logger,
            "SentimentScaleRule output",
            adjusted_size=adjusted,
        )
        return adjusted