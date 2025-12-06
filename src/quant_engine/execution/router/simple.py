from quant_engine.contracts.execution.router import Router
from quant_engine.contracts.execution.order import Order
from .registry import register_router
from quant_engine.utils.logger import get_logger, log_debug


@register_router("SIMPLE")
class SimpleRouter(Router):
    def __init__(self):
        self._logger = get_logger(__name__)

    def route(self, orders, market_data):
        log_debug(self._logger, "SimpleRouter received orders", orders=[o.to_dict() for o in orders])
        # no changes, pass-through router
        log_debug(self._logger, "SimpleRouter pass-through routing complete")
        return orders