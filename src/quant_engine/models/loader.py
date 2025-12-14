# models/loader.py
from typing import Dict, Any
from quant_engine.models.registry import build_model
from quant_engine.utils.logger import get_logger, log_debug


class ModelLoader:
    _logger = get_logger(__name__)

    @staticmethod
    def from_config(model_cfg: Dict[str, Any], symbol: str):
        """
        model_cfg example:
        {
            "type": "PAIR_ZSCORE",
            "params": {"secondary": "ETHUSDT", "lookback": 120}
        }
        """
        name = model_cfg["type"]
        params = model_cfg.get("params", {})

        log_debug(
            ModelLoader._logger,
            "ModelLoader building model",
            name=name,
            symbol=symbol,
            params=params,
        )

        return build_model(name, symbol=symbol, **params)