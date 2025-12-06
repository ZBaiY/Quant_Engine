from quant_engine.contracts.execution.slippage import SlippageModel
from quant_engine.contracts.execution.order import Order, OrderSide, OrderType
from .registry import register_slippage
from quant_engine.utils.logger import get_logger, log_debug


@register_slippage("LINEAR")
class LinearSlippage(SlippageModel):
    def __init__(self, impact=0.0005):
        self.impact = impact
        self._logger = get_logger(__name__)

    def apply(self, order, market_data):
        log_debug(self._logger, "LinearSlippage received order", side=order.side.value, qty=order.qty, impact=self.impact)
        mid = (market_data["bid"] + market_data["ask"]) / 2
        slip = self.impact * order.qty
        log_debug(self._logger, "LinearSlippage computed slippage", mid=mid, slip=slip, adjusted_price=(mid + slip if order.side is OrderSide.BUY else mid - slip))
        return mid + slip if order.side is OrderSide.BUY else mid - slip