from .registry import build_decision
from quant_engine.utils.logger import get_logger, log_debug

class DecisionLoader:
    _logger = get_logger(__name__)
    @staticmethod
    def from_config(cfg: dict, symbol: str):
        """
        cfg example:
        {
            "type": "THRESHOLD",
            "params": {"threshold": 0.0}
        }
        """
        log_debug(DecisionLoader._logger, "DecisionLoader received config", config=cfg)
        name = cfg["type"]
        params = cfg.get("params", {})
        log_debug(DecisionLoader._logger, "DecisionLoader built decision", name=name, params=params)
        return build_decision(name, symbol=symbol, **params)