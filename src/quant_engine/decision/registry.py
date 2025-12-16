# decision/registry.py
from quant_engine.contracts.decision import DecisionBase

DECISION_REGISTRY: dict[str, type[DecisionBase]] = {}

def register_decision(name: str):
    def decorator(cls):
        DECISION_REGISTRY[name] = cls
        return cls
    return decorator


def build_decision(name: str, symbol: str, **kwargs) -> DecisionBase:
    if name not in DECISION_REGISTRY:
        raise ValueError(f"Decision '{name}' not found in registry.")
    return DECISION_REGISTRY[name](symbol=symbol, **kwargs)

from .threshold import *
from .regime import *
from .fusion import *
