# models/ml_model.py
from typing import Dict, Any
from quant_engine.contracts.model import ModelBase, parse_feature_name
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
    # design-time capability requirements inferred from weights
    required_feature_types: set[str] = set()

    def __init__(self, symbol: str, **kwargs):
        """Linear model with weights keyed by feature TYPE or full feature name.

        Expected params (via kwargs):
            weights: Dict[str, float]
            purpose: optional, default "MODEL" (used when weights are keyed by TYPE)
        """
        weights = kwargs.get("weights")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("LinearModel requires non-empty weights={feature_type_or_name: weight}")

        self.purpose = str(kwargs.get("purpose", "MODEL"))
        super().__init__(symbol=symbol, **kwargs)

        # Keep original mapping; keys can be either TYPE (e.g., 'RSI') or full name (e.g., 'RSI_MODEL_BTCUSDT').
        self.weights: Dict[str, float] = {str(k): float(v) for k, v in weights.items()}

        # Design-time capability requirement: infer TYPES from keys.
        inferred_types: set[str] = set()
        for k in self.weights.keys():
            if "_" in k:
                try:
                    t, _p, _s, _r = parse_feature_name(k)
                    inferred_types.add(t)
                except Exception:
                    # If it doesn't parse, treat it as a TYPE-like identifier.
                    inferred_types.add(k)
            else:
                inferred_types.add(k)
        self.required_feature_types = inferred_types

    def predict(self, features: Dict[str, Any]) -> float:
        """Compute linear score.

        If a weight key looks like a full feature name, we use it directly.
        Otherwise we treat it as a feature TYPE and resolve via the semantic index.
        """
        total = 0.0
        for key, weight in self.weights.items():
            try:
                if "_" in key:
                    # Full feature name path
                    value = features.get(key)
                else:
                    # TYPE path
                    value = self.fget(features, ftype=key, purpose=self.purpose)
                if value is None:
                    return 0.0
                total += float(weight) * float(value)
            except Exception:
                return 0.0
        return float(total)