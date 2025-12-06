# strategy/loader.py

from quant_engine.features.loader import FeatureLoader
from quant_engine.models.registry import build_model
from quant_engine.decision.loader import DecisionLoader
from quant_engine.risk.loader import RiskLoader
from quant_engine.execution.loader import ExecutionLoader
from quant_engine.portfolio.loader import PortfolioLoader
from quant_engine.utils.logger import get_logger, log_debug


class StrategyLoader:
    _logger = get_logger(__name__)

    @staticmethod
    def from_config(cfg, data_handler):
        """
        cfg format:
        {
            "features": {...},
            "models": {...},
            "decision": {...},
            "risk": {...},
            "execution": {...},
            "portfolio": {...}
        }
        """

        log_debug(StrategyLoader._logger, "StrategyLoader received config", keys=list(cfg.keys()))

        # Build feature layer
        feature_extractor = FeatureLoader.from_config(cfg["features"], data_handler)

        log_debug(StrategyLoader._logger, "StrategyLoader building models")
        # Build model layer
        models = {
            name: build_model(mcfg["type"], **mcfg.get("params", {}))
            for name, mcfg in cfg["models"].items()
        }

        log_debug(StrategyLoader._logger, "StrategyLoader building decision layer")
        # Decision layer
        decision = DecisionLoader.from_config(cfg["decision"])

        log_debug(StrategyLoader._logger, "StrategyLoader building risk layer")
        # Risk layer
        risk_manager = RiskLoader.from_config(cfg["risk"])

        log_debug(StrategyLoader._logger, "StrategyLoader building execution layer")
        # Execution layer
        execution_engine = ExecutionLoader.from_config(cfg["execution"])

        log_debug(StrategyLoader._logger, "StrategyLoader building portfolio layer")
        # Portfolio layer
        portfolio = PortfolioLoader.from_config(cfg["portfolio"])

        log_debug(StrategyLoader._logger, "StrategyLoader assembled StrategyEngine")
        # Assemble StrategyEngine
        from .engine import StrategyEngine
        return StrategyEngine(
            data_handler=data_handler,
            feature_extractor=feature_extractor,
            models=models,
            decision=decision,
            risk_manager=risk_manager,
            execution_engine=execution_engine,
            portfolio_manager=portfolio
        )
