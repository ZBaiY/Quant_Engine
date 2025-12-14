from typing import Dict, Any
from quant_engine.contracts.model import ModelBase
from .registry import register_model

@register_model("PAIR_ZSCORE")
class PairZScoreModel(ModelBase):
    """
    Minimal pair-trading z-score model.

    Contracts:
    - primary symbol: self.symbol
    - secondary symbol: self.secondary
    - declares secondary feature dependency for resolver
    """

    # required pair feature (resolver-level, symbolic)
    features_secondary = ["SPREAD"]

    def __init__(self, symbol: str, lookback: int = 120, **kwargs):
        super().__init__(symbol=symbol, **kwargs)
        self.lookback = lookback

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Dummy logic:
        - expects a SPREAD feature keyed by BTCUSDT^ETHUSDT or reverse
        - returns sign(z) placeholder
        """
        filtered = self.filter_pair(features)

        # tolerate either ordering
        key1 = f"SPREAD_{self.symbol}^{self.secondary}"
        key2 = f"SPREAD_{self.secondary}^{self.symbol}"

        spread = filtered.get(key1) or filtered.get(key2)
        if spread is None:
            return 0.0

        # placeholder "z-score"
        if spread > 0:
            return -1.0
        elif spread < 0:
            return 1.0
        return 0.0