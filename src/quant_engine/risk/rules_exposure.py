from typing import Dict, Any
from quant_engine.contracts.risk import RiskBase
from quant_engine.risk.registry import register_risk
from quant_engine.utils.logger import get_logger, log_debug


@register_risk("EXPOSURE_LIMIT")
class ExposureLimitRule(RiskBase):
    """
    V4 exposure cap risk rule.

    Contracts:
    - symbol-aware via RiskBase
    - no feature dependency
    - adjust(size, features) -> float
    """

    required_features: list[str] = []

    _logger = get_logger(__name__)

    def __init__(self, symbol: str, limit: float = 3.0, **kwargs):
        super().__init__(symbol=symbol)
        self.limit = limit
        log_debug(
            self._logger,
            "ExposureLimitRule initialized",
            symbol=symbol,
            limit=limit,
        )

    def adjust(self, size: float, features: Dict[str, Any]) -> float:
        log_debug(
            self._logger,
            "ExposureLimitRule adjust() called",
            size=size,
            limit=self.limit,
        )

        if abs(size) > self.limit:
            capped = self.limit * (1 if size > 0 else -1)
            log_debug(
                self._logger,
                "ExposureLimitRule capped",
                capped_size=capped,
            )
            return capped

        log_debug(
            self._logger,
            "ExposureLimitRule passed",
            output=size,
        )
        return size