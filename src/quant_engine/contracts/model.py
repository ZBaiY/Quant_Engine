from typing import Protocol, Dict, Any, List


class ModelProto(Protocol):
    """
    V4 unified model protocol:
        • predict(features) -> float
        • every model must define symbol (primary)
        • may define secondary for pair/stat-arb
        • may define required_features for resolver
    """

    symbol: str
    secondary: str | None
    required_features: List[str]

    def predict(self, features: Dict[str, Any]) -> float:
        ...


# ----------------------------------------------------------------------
# V4 Model Base Class
# ----------------------------------------------------------------------
class ModelBase(ModelProto):
    """
    V4 unified base class for all models used in the engine.

    Responsibilities:
        • store primary symbol (self.symbol)
        • optionally store secondary symbol
        • support feature filtering helpers
        • define required_features (model declares its dependencies)
        • child class must implement predict()
    """

    required_features: List[str] = []     # auto-injected by resolver
    features_secondary: List[str] = []   # for pair/stat-arb models
    symbol: str | None = None             # primary trading symbol
    secondary: str | None = None          # optional secondary

    def __init__(self, symbol: str, **kwargs):
        self.symbol = symbol
        self.secondary = kwargs.get("secondary", None)

        # ------------------------------------------------------------------
        # IMPORTANT:
        #   - required_features / features_secondary are DECLARED at class level
        #   - we must copy them to the instance before expanding
        #   - otherwise models contaminate each other via shared mutable lists
        # ------------------------------------------------------------------

        base_required = list(type(self).required_features)
        base_secondary = list(type(self).features_secondary)
        # expand primary-symbol features
        self.required_features = [
            f"{r}_{self.symbol}" for r in base_required
        ]

        # expand secondary / pair features
        if self.secondary:
            self.features_secondary = [
                f"{r}_{self.secondary}^{self.symbol}" for r in base_secondary
            ]
        else:
            self.features_secondary = []
        
        
        

    # ------------------------------------------------------------------
    # Feature filtering helpers (core of v4 architecture)
    # ------------------------------------------------------------------
    def filter_symbol(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Filter features for primary symbol only."""
        suffix = f"_{self.symbol}"
        return {k: v for k, v in features.items() if k.endswith(suffix)}

    def filter_pair(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Filter features for primary + secondary symbol + pair features."""
        if not self.secondary:
            return self.filter_symbol(features)

        primary_suffix = f"_{self.symbol}"
        secondary_suffix = f"_{self.secondary}"
        pair_key_1 = f"{self.symbol}^{self.secondary}"
        pair_key_2 = f"{self.secondary}^{self.symbol}"

        out = {}
        for k, v in features.items():
            if (
                k.endswith(primary_suffix)
                or k.endswith(secondary_suffix)
                or pair_key_1 in k
                or pair_key_2 in k
            ):
                out[k] = v
        return out

    # ------------------------------------------------------------------
    # Child models must implement this
    # ------------------------------------------------------------------
    def predict(self, features: Dict[str, Any]) -> float:
        raise NotImplementedError("Model must implement predict()")