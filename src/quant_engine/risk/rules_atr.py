from quant_engine.contracts.risk import RiskProto
from .registry import register_risk
from quant_engine.utils.logger import get_logger, log_debug

@register_risk("ATR_SIZE")
class ATRSizeRule(RiskProto):
    _logger = get_logger(__name__)

    def __init__(self, base_size=1.0, atr_key="atr"):
        self.base_size = base_size
        self.atr_key = atr_key
        log_debug(self._logger, "ATRSizeRule initialized", base_size=base_size, atr_key=atr_key)

    def adjust(self, size: float, features: dict) -> float:
        log_debug(self._logger, "ATRSizeRule adjust() called", size=size, atr=features.get(self.atr_key))
        atr = features.get(self.atr_key, 1.0)
        result = size * self.base_size / (atr + 1e-8)
        log_debug(self._logger, "ATRSizeRule output", adjusted_size=result)
        return result