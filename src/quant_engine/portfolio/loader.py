from .registry import build_portfolio
from quant_engine.utils.logger import get_logger, log_debug

class PortfolioLoader:
    _logger = get_logger(__name__)
    @staticmethod
    def from_config(cfg: dict):
        log_debug(PortfolioLoader._logger, "PortfolioLoader received config", config=cfg)
        name = cfg["type"]
        params = cfg.get("params", {})
        log_debug(PortfolioLoader._logger, "PortfolioLoader built portfolio", name=name, params=params)
        return build_portfolio(name, **params)