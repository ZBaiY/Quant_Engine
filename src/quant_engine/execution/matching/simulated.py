from quant_engine.contracts.execution.matching import MatchingEngine
from .registry import register_matching
from quant_engine.utils.logger import get_logger, log_debug, log_info


@register_matching("SIMULATED")
class SimulatedMatchingEngine(MatchingEngine):
    def __init__(self):
        self._logger = get_logger(__name__)

    def execute(self, order, market_data):
        log_debug(self._logger, "SimulatedMatchingEngine received order", side=order.side, qty=order.qty, order_type=order.order_type, price=order.price)
        fill_price = market_data["mid"]
        fee = abs(order.qty) * 0.0004
        log_info(self._logger, "SimulatedMatchingEngine produced fill", fill_price=fill_price, filled_qty=order.qty, fee=fee)
        return {
            "fill_price": fill_price,
            "filled_qty": order.qty,
            "fee": fee,
            "slippage": fill_price - (market_data["mid"])
        }