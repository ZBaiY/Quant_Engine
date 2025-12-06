# decision/threshold.py
from quant_engine.contracts.decision import DecisionProto
from .registry import register_decision
from quant_engine.utils.logger import get_logger, log_debug


@register_decision("THRESHOLD")
class ThresholdDecision(DecisionProto):
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self._logger = get_logger(__name__)
        log_debug(self._logger, "ThresholdDecision initialized", threshold=threshold)

    def decide(self, context) -> float:
        score = context.get("score", 0.0)
        log_debug(self._logger, "ThresholdDecision received score", score=score)
        decision = 1.0 if score > self.threshold else -1.0
        log_debug(self._logger, "ThresholdDecision output", decision=decision)
        return decision