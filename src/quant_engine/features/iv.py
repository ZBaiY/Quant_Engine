# src/quant_engine/features/iv.py
import pandas as pd
from quant_engine.contracts.feature import FeatureChannel
from .registry import register_feature
from quant_engine.utils.logger import get_logger, log_debug


@register_feature("IV30")
class IV30Feature(FeatureChannel):
    _logger = get_logger(__name__)
    """Implied Volatility 30d."""
    def compute(self, df: pd.DataFrame):
        log_debug(self._logger, "IV30Feature compute() called")
        if "iv_30d" not in df:
            result = {"iv30": None}
            log_debug(self._logger, "IV30Feature output", value=None)
            return result
        result = {"iv30": float(df["iv_30d"].iloc[-1])}
        log_debug(self._logger, "IV30Feature output", value=result["iv30"])
        return result


@register_feature("IV_SKEW")
class IVSkewFeature(FeatureChannel):
    _logger = get_logger(__name__)
    """25d call - 25d put skew."""
    def compute(self, df: pd.DataFrame):
        log_debug(self._logger, "IVSkewFeature compute() called")
        if "iv_25d_call" not in df or "iv_25d_put" not in df:
            result = {"iv_skew": None}
            log_debug(self._logger, "IVSkewFeature output", value=None)
            return result
        skew = df["iv_25d_call"].iloc[-1] - df["iv_25d_put"].iloc[-1]
        result = {"iv_skew": float(skew)}
        log_debug(self._logger, "IVSkewFeature output", value=result["iv_skew"])
        return result