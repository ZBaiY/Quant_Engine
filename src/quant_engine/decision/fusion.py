# decision/fusion.py
from quant_engine.contracts.decision import DecisionBase
from .registry import register_decision
from quant_engine.utils.logger import get_logger, log_debug


@register_decision("FUSION")
class FusionDecision(DecisionBase):
    """
    Combine multiple model scores:
        score_final = w1 * model1 + w2 * model2 + ...
    """
    def __init__(self, weights: dict[str, float]):
        self.weights = weights
        self._logger = get_logger(__name__)
        log_debug(self._logger, "FusionDecision initialized", weights=weights)

    def decide(self, context: dict) -> float:
        log_debug(self._logger, "FusionDecision received context", context=context)
        # context: {"model_score": ..., "sent_score": ..., ...}
        total = 0.0
        for key, w in self.weights.items():
            total += w * context.get(key, 0.0)
        log_debug(self._logger, "FusionDecision computed fused score", score=total)
        return total