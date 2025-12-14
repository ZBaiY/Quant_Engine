# models/regime.py
from typing import Dict, Any
from quant_engine.contracts.model import ModelBase
from quant_engine.models.registry import register_model


@register_model("VOL_REGIME")
class VolRegimeModel(ModelBase):
    """
    V4 volatility regime classifier.

    Contracts:
    - symbol-aware via ModelBase
    - declares VOLATILITY dependency
    - predict(features) -> float
    """

    required_features = ["VOLATILITY"]

    def __init__(self, symbol: str, threshold: float = 0.02, **kwargs):
        super().__init__(symbol=symbol, **kwargs)
        self.threshold = threshold

    def predict(self, features: Dict[str, Any]) -> float:
        filtered = self.filter_symbol(features)
        key = f"VOLATILITY_{self.symbol}"
        vol = filtered.get(key, 0.0)
        return 1.0 if vol > self.threshold else -1.0