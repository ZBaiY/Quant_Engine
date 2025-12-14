MATCHING_REGISTRY = {}

def register_matching(name: str):
    def decorator(cls):
        MATCHING_REGISTRY[name] = cls
        return cls
    return decorator

def build_matching(name: str, symbol: str, **kwargs):
    return MATCHING_REGISTRY[name](symbol=symbol, **kwargs)

from .live import *
from .simulated import *