# risk/engine.py
from quant_engine.contracts.risk import RiskBase
from quant_engine.utils.logger import get_logger, log_debug

class RiskEngine:
    _logger = get_logger(__name__)

    def __init__(self, rules: list[RiskBase]):
        self.rules = rules
        log_debug(self._logger, "RiskEngine initialized", rule_count=len(rules))

    def apply(self, size: float, features: dict) -> float:
        log_debug(self._logger, "RiskEngine apply() called", initial_size=size)
        for rule in self.rules:
            log_debug(self._logger, "RiskEngine applying rule", rule=rule.__class__.__name__, size_before=size)
            size = rule.adjust(size, features)
            log_debug(self._logger, "RiskEngine rule output", rule=rule.__class__.__name__, size_after=size)
        log_debug(self._logger, "RiskEngine final size", size=size)
        return size