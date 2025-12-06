from quant_engine.utils.logger import get_logger, log_debug

class ATRSizer:
    _logger = get_logger(__name__)

    def __init__(self, risk_fraction=0.02):
        self.risk_fraction = risk_fraction
        log_debug(self._logger, "ATRSizer initialized", risk_fraction=risk_fraction)

    def size(self, intent, volatility=1.0):
        log_debug(self._logger, "ATRSizer size() called", intent=intent, volatility=volatility)
        result = intent * self.risk_fraction / volatility
        log_debug(self._logger, "ATRSizer output", size=result)
        return result