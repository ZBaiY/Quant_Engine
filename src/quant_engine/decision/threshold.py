# decision/threshold.py
from quant_engine.contracts.decision import DecisionBase
from .registry import register_decision


@register_decision("THRESHOLD")
class ThresholdDecision(DecisionBase):
    def __init__(
        self,
        symbol: str | None = None,
        threshold: float = 0.0,
        score_key: str = "main",
        **kwargs,
    ):
        super().__init__(symbol=symbol, **kwargs)
        self.threshold = float(threshold)
        self.score_key = str(score_key)

    def decide(self, context: dict) -> float:
        # Prefer model outputs (v4 context shape). Fallback to legacy flat keys.
        models = context.get("models")
        if not isinstance(models, dict):
            models = {}
        score = models.get(self.score_key)
        if score is None:
            score = context.get("score", 0.0)
        x = float(score)
        return 1.0 if x > self.threshold else -1.0
    
@register_decision("ZSCORE_THRESHOLD")
class ZScoreThresholdDecision(DecisionBase):
    """
    V4 Z-score threshold decision with hysteresis.

    Contracts:
    - symbol-agnostic
    - operates primarily on ZSCORE feature from context['features'] (fallback: model outputs)
    - decide(context) -> float in {-1.0, 0.0, +1.0}
    """
    # design-time capability requirement
    required_feature_types = {"ZSCORE"}

    def __init__(
        self,
        symbol: str | None = None,
        enter: float = 2.0,
        exit: float = 0.5,
        purpose: str = "DECISION",
        **kwargs,
    ):
        super().__init__(symbol=symbol, **kwargs)
        self.enter = float(enter)
        self.exit = float(exit)
        self.purpose = str(purpose)
        self._position = 0.0  # internal state: -1, 0, +1

    def decide(self, context: dict) -> float:
        # Prefer ZSCORE feature (explicit feature dependency). Fallback to model outputs.
        score: float
        features = context.get("features")
        if isinstance(features, dict):
            try:
                # DecisionBase.fget requires symbol; prefer bound self.symbol.
                if self.symbol is None:
                    raise ValueError("ZScoreThresholdDecision requires symbol=... to resolve ZSCORE feature")
                score = float(self.fget(features, ftype="ZSCORE", purpose=self.purpose, symbol=self.symbol))
            except Exception:
                score = 0.0
        else:
            score = 0.0

        if score == 0.0:
            models = context.get("models")
            if isinstance(models, dict):
                # tolerate multiple key conventions
                v = models.get("zscore")
                if v is None:
                    v = models.get("main")
                if v is not None:
                    score = float(v)
            if score == 0.0:
                score = float(context.get("model_score", 0.0))

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

        return float(self._position)