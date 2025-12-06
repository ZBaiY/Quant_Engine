from quant_engine.contracts.risk import RiskProto
from quant_engine.risk.registry import register_risk
from quant_engine.utils.logger import get_logger, log_debug

@register_risk("STOP_LOSS")
class StopLossRule(RiskProto):
    _logger = get_logger(__name__)

    def __init__(self, key="pnl", max_loss=-0.05):
        self.key = key
        self.max_loss = max_loss
        log_debug(self._logger, "StopLossRule initialized", key=key, max_loss=max_loss)

    def adjust(self, size: float, features: dict) -> float:
        log_debug(self._logger, "StopLossRule adjust() called", size=size, pnl=features.get(self.key))
        pnl = features.get(self.key, 0.0)
        if pnl < self.max_loss:
            log_debug(self._logger, "StopLossRule triggered", output=0.0)
            return 0.0
        log_debug(self._logger, "StopLossRule passed", output=size)
        return size