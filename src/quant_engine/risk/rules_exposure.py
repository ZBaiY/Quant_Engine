from quant_engine.contracts.risk import RiskProto
from quant_engine.risk.registry import register_risk
from quant_engine.utils.logger import get_logger, log_debug

@register_risk("EXPOSURE_LIMIT")
class ExposureLimitRule(RiskProto):
    _logger = get_logger(__name__)

    def __init__(self, limit=3.0):
        self.limit = limit
        log_debug(self._logger, "ExposureLimitRule initialized", limit=limit)

    def adjust(self, size: float, features: dict) -> float:
        log_debug(self._logger, "ExposureLimitRule adjust() called", size=size, limit=self.limit)
        if abs(size) > self.limit:
            log_debug(self._logger, "ExposureLimitRule capped", capped_size=self.limit * (1 if size > 0 else -1))
            return self.limit * (1 if size > 0 else -1)
        log_debug(self._logger, "ExposureLimitRule passed", output=size)
        return size