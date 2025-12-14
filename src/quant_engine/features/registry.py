# features/registry.py
from quant_engine.contracts.feature import FeatureChannel

### Feature registry and builder, this is because we have many features.

# Global registry
FEATURE_REGISTRY: dict[str, type] = {}


def register_feature(name: str):
    """
    Decorator: @register_feature("RSI")
    Automatically registers class into FEATURE_REGISTRY.
    """
    def decorator(cls):
        FEATURE_REGISTRY[name] = cls
        return cls
    return decorator


def build_feature(name: str, symbol=None, **params) -> FeatureChannel:
    """Instantiate a feature class by name (multi‑symbol ready)."""
    if name not in FEATURE_REGISTRY:
        raise ValueError(f"Feature '{name}' not found in registry.")
    
    cls = FEATURE_REGISTRY[name]

    return cls(symbol=symbol, **params)


from quant_engine.features.ta.ta import *                 # noqa: F401,F403
from quant_engine.features.volatility.volatility import * # noqa: F401,F403
from quant_engine.features.microstructure.microstructure import *  # noqa: F401,F403
from quant_engine.features.options.iv import *            # noqa: F401,F403
from quant_engine.features.options.iv_surface import *    # noqa: F401,F403
