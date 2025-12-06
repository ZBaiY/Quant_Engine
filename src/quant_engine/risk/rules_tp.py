from quant_engine.contracts.risk import RiskProto
from quant_engine.risk.registry import register_risk
from quant_engine.utils.logger import get_logger, log_debug

@register_risk("TAKE_PROFIT")
class TakeProfitRule(RiskProto):
    _logger = get_logger(__name__)

    def __init__(self, max_gain=0.1):
        self.max_gain = max_gain
        log_debug(self._logger, "TakeProfitRule initialized", max_gain=max_gain)

    def size(self, intent: float, volatility: float = 1.0) -> float:
        log_debug(self._logger, "TakeProfitRule size() called", intent=intent, max_gain=self.max_gain)
        if intent > self.max_gain:
            log_debug(self._logger, "TakeProfitRule triggered", output=0.0)
            return 0.0
        log_debug(self._logger, "TakeProfitRule passed", output=intent)
        return intent