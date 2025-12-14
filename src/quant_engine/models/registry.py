# models/registry.py
from quant_engine.contracts.model import ModelBase
# src/quant_engine/models/__init__.py


MODEL_REGISTRY: dict[str, type[ModelBase]] = {}

def register_model(name: str):
    """Decorator: @register_model("OU_MODEL")"""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(name: str, **kwargs) -> ModelBase:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry.")
    return MODEL_REGISTRY[name](**kwargs)


from .momentum import *
from .statistical import *
from .regime import *
from .ml_model import *
from .physics import *