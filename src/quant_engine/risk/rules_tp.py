from typing import Dict, Any
from quant_engine.contracts.risk import RiskBase
from quant_engine.risk.registry import register_risk
from quant_engine.utils.logger import get_logger, log_debug


@register_risk("TAKE_PROFIT")
class TakeProfitRule(RiskBase):
    """
    V4 take-profit risk rule.

    Contracts:
    - symbol-aware via RiskBase
    - no hard feature dependency (uses runtime PnL / return key)
    - adjust(size, features) -> float
    """

    required_features: list[str] = []

    _logger = get_logger(__name__)

    def __init__(
        self,
        symbol: str,
        key: str = "pnl",
        max_gain: float = 0.1,
        **kwargs,
    ):
        super().__init__(symbol=symbol)
        self.key = key
        self.max_gain = max_gain
        log_debug(
            self._logger,
            "TakeProfitRule initialized",
            symbol=symbol,
            key=key,
            max_gain=max_gain,
        )

    def adjust(self, size: float, features: Dict[str, Any]) -> float:
        value = features.get(self.key, 0.0)

        log_debug(
            self._logger,
            "TakeProfitRule adjust() called",
            size=size,
            value=value,
        )

        if value > self.max_gain:
            log_debug(
                self._logger,
                "TakeProfitRule triggered",
                output=0.0,
            )
            return 0.0

        log_debug(
            self._logger,
            "TakeProfitRule passed",
            output=size,
        )
        return size