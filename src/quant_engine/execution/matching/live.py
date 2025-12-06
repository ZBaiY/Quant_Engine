# execution/matching/live.py
from quant_engine.contracts.execution.matching import MatchingEngine
from .registry import register_matching
from quant_engine.utils.logger import get_logger, log_debug, log_info


@register_matching("LIVE_BINANCE")
class LiveBinanceMatchingEngine(MatchingEngine):
    def __init__(self, client):
        self.client = client
        self._logger = get_logger(__name__)

    def execute(self, order, market_data):
        """
        send order to exchange
        wait for fill
        return fill info
        """
        log_debug(self._logger, "LiveBinanceMatchingEngine received order", side=order.side, qty=order.qty, order_type=order.order_type, price=order.price)
        # placeholder
        log_info(self._logger, "LiveBinanceMatchingEngine executed live order (placeholder)", result={"fill_price": None, "filled_qty": 0, "fee": 0, "slippage": 0})
        return {
            "fill_price": None,
            "filled_qty": 0,
            "fee": 0,
            "slippage": 0
        }