# models/ml_model.py
from typing import Dict, Any
from quant_engine.contracts.model import ModelBase
from quant_engine.models.registry import register_model


@register_model("LINEAR")
class LinearModel(ModelBase):
    """
    V4 linear model.

    Contracts:
    - symbol-aware via ModelBase
    - declares required_features from weight keys
    - predict(features) -> float
    """
    required_features: list[str] = []

    def __init__(self, symbol: str, weights: Dict[str, float], **kwargs):
        super().__init__(symbol=symbol, **kwargs)
        self.weights = weights
        # declare dependencies for resolver (feature base names)
        self.required_features = [k.split("_")[0] for k in weights.keys()]

    def predict(self, features: Dict[str, Any]) -> float:
        filtered = self.filter_symbol(features)
        return sum(
            w * filtered.get(k, 0.0)
            for k, w in self.weights.items()
        )