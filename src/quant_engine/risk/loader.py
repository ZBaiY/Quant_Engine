# risk/loader.py
from .registry import build_risk
from .engine import RiskEngine
from quant_engine.utils.logger import get_logger, log_debug

class RiskLoader:
    _logger = get_logger(__name__)

    @staticmethod
    def from_config(cfg: dict):
        """
        Create a RiskEngine from configuration.
        """
        log_debug(RiskLoader._logger, "RiskLoader received config", config=cfg)
        rules_cfg = cfg.get("rules", {})
        rules = [build_risk(name, **params) for name, params in rules_cfg.items()]
        log_debug(RiskLoader._logger, "RiskLoader built RiskEngine", rule_count=len(rules))
        return RiskEngine(rules)