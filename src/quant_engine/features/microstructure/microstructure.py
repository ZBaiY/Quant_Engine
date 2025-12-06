# src/quant_engine/features/microstructure/microstructure.py
# Microstructure feature channels (L2/L3 orderbook derived)
import pandas as pd
from quant_engine.contracts.feature import FeatureChannel
from ..registry import register_feature
from quant_engine.utils.logger import get_logger, log_debug


@register_feature("SPREAD")
class SpreadFeature(FeatureChannel):
    """Best bid/ask spread."""
    _logger = get_logger(__name__)

    def compute(self, df: pd.DataFrame):
        # df is the realtime OHLCV/orderbook window. Microstructure features rely on bid/ask and size fields.
        log_debug(self._logger, "SpreadFeature compute() called")
        spread = df["ask"] - df["bid"]
        result = {"spread": float(spread.iloc[-1])}
        log_debug(self._logger, "SpreadFeature output", value=result["spread"])
        return result


@register_feature("IMBALANCE")
class OrderImbalanceFeature(FeatureChannel):
    """Orderbook imbalance = (bid_size - ask_size) / (sum)."""
    _logger = get_logger(__name__)

    def compute(self, df: pd.DataFrame):
        # df is the realtime OHLCV/orderbook window. Microstructure features rely on bid/ask and size fields.
        log_debug(self._logger, "OrderImbalanceFeature compute() called")
        num = df["bid_size"] - df["ask_size"]
        den = df["bid_size"] + df["ask_size"] + 1e-9
        imbalance = num / den
        result = {"order_imbalance": float(imbalance.iloc[-1])}
        log_debug(self._logger, "OrderImbalanceFeature output", value=result["order_imbalance"])
        return result