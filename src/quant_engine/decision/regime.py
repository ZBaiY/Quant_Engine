# decision/regime.py
from quant_engine.contracts.decision import DecisionBase
from .registry import register_decision
from quant_engine.utils.logger import get_logger, log_debug


@register_decision("REGIME")
class RegimeDecision(DecisionBase):
    def __init__(self, bull: float = 1.0, bear: float = -1.0):
        self.bull = bull
        self.bear = bear
        self._logger = get_logger(__name__)
        log_debug(self._logger, "RegimeDecision initialized", bull=bull, bear=bear)

    def decide(self, context) -> float:
        regime_label = context.get("regime_label", 0)
        log_debug(self._logger, "RegimeDecision received regime_label", regime_label=regime_label)
        if regime_label > 0:
            log_debug(self._logger, "RegimeDecision output", decision=self.bull)
            return self.bull
        elif regime_label < 0:
            log_debug(self._logger, "RegimeDecision output", decision=self.bear)
            return self.bear
        log_debug(self._logger, "RegimeDecision output", decision=0.0)
        return 0.0