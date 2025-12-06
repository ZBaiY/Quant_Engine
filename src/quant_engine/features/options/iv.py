# src/quant_engine/features/options/iv.py
# NOTE: These IV features use OHLCV-derived IV columns.
# Option-chain–based IV surface features live in iv_surface.py.
from quant_engine.contracts.feature import FeatureChannel
from ..registry import register_feature
from quant_engine.utils.logger import get_logger, log_debug


@register_feature("IV30")
class IV30Feature(FeatureChannel):
    def __init__(self, symbol=None, **kwargs):
        self.symbol = symbol
    _logger = get_logger(__name__)
    """Implied Volatility 30d."""
    def compute(self, context):
        # context is the full context dictionary; option-chain features ignore context, but these IV metrics use the OHLCV DataFrame in context.
        df = context.get("ohlcv")
        log_debug(self._logger, "IV30Feature compute() called")
        if df is None or "iv_30d" not in df:
            result = {"iv30": None}
            log_debug(self._logger, "IV30Feature output", value=None)
            return result
        result = {"iv30": float(df["iv_30d"].iloc[-1])}
        log_debug(self._logger, "IV30Feature output", value=result["iv30"])
        return result


@register_feature("IV_SKEW")
class IVSkewFeature(FeatureChannel):
    def __init__(self, symbol=None, **kwargs):
        self.symbol = symbol
    _logger = get_logger(__name__)
    """25d call - 25d put skew."""
    def compute(self, context):
        # context is the full context dictionary; option-chain features ignore context, but these IV metrics use the OHLCV DataFrame in context.
        df = context.get("ohlcv")
        log_debug(self._logger, "IVSkewFeature compute() called")
        if df is None or "iv_25d_call" not in df or "iv_25d_put" not in df:
            result = {"iv_skew": None}
            log_debug(self._logger, "IVSkewFeature output", value=None)
            return result
        skew = df["iv_25d_call"].iloc[-1] - df["iv_25d_put"].iloc[-1]
        result = {"iv_skew": float(skew)}
        log_debug(self._logger, "IVSkewFeature output", value=result["iv_skew"])
        return result