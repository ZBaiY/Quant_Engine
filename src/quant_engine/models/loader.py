# models/loader.py
from typing import Dict, Any
from quant_engine.models.registry import build_model


class ModelLoader:

    @staticmethod
    def from_config(model_cfg: Dict[str, Any], symbol: str) -> Any:
        """
        model_cfg example:
        {
            "type": "PAIR_ZSCORE",
            "params": {"secondary": "ETHUSDT", "lookback": 120}
        }
        """
        name = model_cfg["type"]
        params = model_cfg.get("params", {})

        return build_model(name, symbol=symbol, **params)