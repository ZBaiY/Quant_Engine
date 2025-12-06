# execution/slippage/depth.py
from quant_engine.contracts.execution.slippage import SlippageModel
from .registry import register_slippage
from quant_engine.utils.logger import get_logger, log_debug


@register_slippage("DEPTH")
class DepthSlippage(SlippageModel):
    def __init__(self, depth_key="depth"):
        self.depth_key = depth_key
        self._logger = get_logger(__name__)

    def apply(self, order, market_data):
        log_debug(self._logger, "DepthSlippage received order", side=order.side, qty=order.qty)
        depth = market_data[self.depth_key]  # e.g. dict with volume levels
        # placeholder: realistic depth = complex model
        adjusted_price = market_data["mid"] + (order.qty / (depth + 1e-8))
        log_debug(self._logger, "DepthSlippage computed slippage", depth=depth, mid=market_data["mid"], adjusted_price=adjusted_price)
        return adjusted_price