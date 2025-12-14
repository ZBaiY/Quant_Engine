ROUTER_REGISTRY = {}

def register_router(name: str):
    def decorator(cls):
        ROUTER_REGISTRY[name] = cls
        return cls
    return decorator

def build_router(name: str, symbol: str, **kwargs):
    return ROUTER_REGISTRY[name](symbol=symbol, **kwargs)

from .l1_aware import *
from .simple import *
