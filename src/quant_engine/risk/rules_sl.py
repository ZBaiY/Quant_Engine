from typing import Dict, Any
from quant_engine.contracts.risk import RiskBase
from quant_engine.risk.registry import register_risk
from quant_engine.utils.logger import get_logger, log_debug


@register_risk("STOP_LOSS")
class StopLossRule(RiskBase):
    """
    V4 stop-loss risk rule.

    Contracts:
    - symbol-aware via RiskBase
    - no hard feature dependency (uses runtime PnL key)
    - adjust(size, features) -> float
    """

    required_features: list[str] = []

    _logger = get_logger(__name__)

    def __init__(
        self,
        symbol: str,
        key: str = "pnl",
        max_loss: float = -0.05,
        **kwargs,
    ):
        super().__init__(symbol=symbol)
        self.key = key
        self.max_loss = max_loss
        log_debug(
            self._logger,
            "StopLossRule initialized",
            symbol=symbol,
            key=key,
            max_loss=max_loss,
        )

    def adjust(self, size: float, features: Dict[str, Any]) -> float:
        pnl = features.get(self.key, 0.0)

        log_debug(
            self._logger,
            "StopLossRule adjust() called",
            size=size,
            pnl=pnl,
        )

        if pnl < self.max_loss:
            log_debug(
                self._logger,
                "StopLossRule triggered",
                output=0.0,
            )
            return 0.0

        log_debug(
            self._logger,
            "StopLossRule passed",
            output=size,
        )
        return size