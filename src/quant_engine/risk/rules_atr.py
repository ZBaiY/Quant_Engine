from typing import Any, Dict
from quant_engine.contracts.risk import RiskBase
from .registry import register_risk
from quant_engine.utils.logger import get_logger, log_debug

@register_risk("ATR_SIZER")
class ATRSizer(RiskBase):
    """
    V4 ATR-based position sizer.

    Contracts:
    - symbol-aware via RiskBase
    - declares ATR dependency for feature resolver
    - adjust(size, features) -> float
    """

    required_features = ["ATR"]

    _logger = get_logger(__name__)

    def __init__(self, symbol: str, risk_fraction: float = 0.02, **kwargs):
        super().__init__(symbol=symbol)
        self.risk_fraction = risk_fraction
        log_debug(
            self._logger,
            "ATRSizer initialized",
            symbol=symbol,
            risk_fraction=risk_fraction,
        )

    def adjust(self, size: float, features: Dict[str, Any]) -> float:
        """
        Scale position size inversely with ATR.
        """
        filtered = self.filter_symbol(features)
        atr_key = f"ATR_{self.symbol}"
        atr = filtered.get(atr_key, 1.0)

        log_debug(
            self._logger,
            "ATRSizer adjust() called",
            size=size,
            atr=atr,
        )

        adjusted = size * self.risk_fraction / max(atr, 1e-8)

        log_debug(
            self._logger,
            "ATRSizer output",
            adjusted_size=adjusted,
        )
        return adjusted