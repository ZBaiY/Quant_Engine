# decision/threshold.py
from quant_engine.contracts.decision import DecisionBase
from .registry import register_decision
from quant_engine.utils.logger import get_logger, log_debug


@register_decision("THRESHOLD")
class ThresholdDecision(DecisionBase):
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
    
@register_decision("ZSCORE_THRESHOLD")
class ZScoreThresholdDecision(DecisionBase):
    """
    V4 Z-score threshold decision with hysteresis.

    Contracts:
    - symbol-agnostic
    - operates purely on model_score from context
    - decide(context) -> float in {-1.0, 0.0, +1.0}
    """

    def __init__(self, enter: float = 2.0, exit: float = 0.5, **kwargs):
        super().__init__()
        self.enter = float(enter)
        self.exit = float(exit)
        self._position = 0.0  # internal state: -1, 0, +1
        self._logger = get_logger(__name__)
        log_debug(
            self._logger,
            "ZScoreThresholdDecision initialized",
            enter=self.enter,
            exit=self.exit,
        )

    def decide(self, context):
        score = context.get("model_score", 0.0)

        log_debug(
            self._logger,
            "ZScoreThresholdDecision received score",
            score=score,
            position=self._position,
        )

        # enter logic
        if self._position == 0.0:
            if score >= self.enter:
                self._position = -1.0
            elif score <= -self.enter:
                self._position = 1.0

        # exit logic
        elif self._position > 0 and score >= -self.exit:
            self._position = 0.0
        elif self._position < 0 and score <= self.exit:
            self._position = 0.0

        log_debug(
            self._logger,
            "ZScoreThresholdDecision output",
            decision=self._position,
        )
        return self._position