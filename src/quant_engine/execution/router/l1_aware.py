from quant_engine.contracts.execution.router import Router
from quant_engine.contracts.execution.order import Order
from .registry import register_router
from quant_engine.utils.logger import get_logger, log_debug


@register_router("L1_AWARE")
class L1AwareRouter(Router):
    def __init__(self):
        self._logger = get_logger(__name__)

    def route(self, orders, market_data):
        log_debug(self._logger, "L1AwareRouter received orders", orders=[o.to_dict() for o in orders])
        bid = market_data["bid"]
        ask = market_data["ask"]
        log_debug(self._logger, "L1AwareRouter market data", bid=bid, ask=ask)

        routed = []
        for o in orders:
            if o.order_type == "LIMIT" and o.price is None:
                if o.side == "BUY":
                    o.price = bid
                else:
                    o.price = ask
            log_debug(self._logger, "L1AwareRouter routed order", side=o.side, order_type=o.order_type, price=o.price)
            routed.append(o)
        return routed