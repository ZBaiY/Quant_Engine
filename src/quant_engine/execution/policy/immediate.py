from quant_engine.contracts.execution.policy import ExecutionPolicy
from quant_engine.contracts.execution.order import Order
from .registry import register_policy
from quant_engine.utils.logger import get_logger, log_debug


@register_policy("IMMEDIATE")
class ImmediatePolicy(ExecutionPolicy):
    def __init__(self):
        self._logger = get_logger(__name__)

    def generate(self, target_position, portfolio_state, market_data):
        log_debug(self._logger, "ImmediatePolicy received target_position", target_position=target_position)
        current_pos = portfolio_state.get("position", 0)
        diff = target_position - current_pos
        
        if diff == 0:
            return []

        side = "BUY" if diff > 0 else "SELL"
        qty = abs(diff)

        log_debug(self._logger, "ImmediatePolicy generated order", side=side, qty=qty)

        return [
            Order(
                side=side,
                qty=qty,
                order_type="MARKET",
                price=None,
                tag="immediate"
            )
        ]