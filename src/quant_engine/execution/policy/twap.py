import math
from decimal import Decimal, ROUND_FLOOR

from quant_engine.contracts.execution.policy import PolicyBase
from quant_engine.contracts.execution.order import (
    Order,
    OrderSide,
    OrderType,
)
from .registry import register_policy
from quant_engine.utils.logger import get_logger, log_debug


@register_policy("TWAP")
class TWAPPolicy(PolicyBase):
    def __init__(self, symbol: str,slices=5):
        self.symbol = symbol
        self.slices = slices
        self._logger = get_logger(__name__)

    def _get_price_ref(self, market_data):
        if not market_data:
            return None
        orderbook = market_data.get("orderbook")
        if orderbook is not None:
            bid = orderbook.get_attr("best_bid") if hasattr(orderbook, "get_attr") else None
            ask = orderbook.get_attr("best_ask") if hasattr(orderbook, "get_attr") else None
            if bid is not None and ask is not None:
                return (float(bid) + float(ask)) / 2.0
            mid = orderbook.get_attr("mid") if hasattr(orderbook, "get_attr") else None
            if mid is not None:
                return float(mid)
        ohlcv = market_data.get("ohlcv")
        if ohlcv is not None:
            close = ohlcv.get_attr("close") if hasattr(ohlcv, "get_attr") else None
            if close is not None:
                return float(close)
        return None

    def _d(self, value):
        return Decimal(str(value))

    def _lots_from_qty(self, qty, step_size):
        qty_d = self._d(qty)
        if qty_d < 0:
            qty_d = -qty_d
        if qty_d <= 0:
            return 0
        lots = (qty_d / step_size).to_integral_value(rounding=ROUND_FLOOR)
        return int(lots)

    def _qty_from_lots(self, lots, step_size):
        return step_size * Decimal(int(lots))

    def _get_conservative_buy_price(self, market_data, price_ref, slippage_bps):
        orderbook = market_data.get("orderbook") if market_data else None
        ask = orderbook.get_attr("best_ask") if orderbook is not None and hasattr(orderbook, "get_attr") else None
        if ask is not None:
            try:
                return max(float(ask), float(price_ref))
            except (TypeError, ValueError):
                pass
        buffer = max(0.0, float(slippage_bps)) / 1e4
        return float(price_ref) * (1.0 + buffer)

    def _fee_buffer(self, min_notional, portfolio_state):
        fee_buffer = float(portfolio_state.get("fee_buffer", 0.0))
        if fee_buffer > 0.0:
            return fee_buffer
        return 0.0

    def generate(self, target_position, portfolio_state, market_data):
        log_debug(self._logger, "TWAPPolicy received target_position", target_position=target_position, slices=self.slices)
        price_ref = self._get_price_ref(market_data)
        if price_ref is None or price_ref <= 0:
            return []

        cash = float(portfolio_state.get("cash", 0.0))
        step_size = self._d(portfolio_state.get("qty_step", portfolio_state.get("step_size", 1)))
        min_qty = float(portfolio_state.get("min_qty", 0.0))
        min_notional = float(portfolio_state.get("min_notional", 0.0))
        slippage_bps = float(portfolio_state.get("slippage_bps", 0.0))
        current_position_qty = float(
            portfolio_state.get("position_qty", portfolio_state.get("position", 0.0))
        )
        current_lots = portfolio_state.get("position_lots")
        if current_lots is None:
            current_lots = self._lots_from_qty(current_position_qty, step_size)
        current_lots = int(current_lots)
        equity = float(portfolio_state.get("total_equity", cash + current_position_qty * price_ref))
        if equity <= 0:
            return []

        desired_notional = float(target_position) * equity
        desired_qty = desired_notional / price_ref
        desired_lots = self._lots_from_qty(desired_qty, step_size)
        if desired_lots > current_lots and cash > 0.0:
            conservative_price = self._get_conservative_buy_price(market_data, price_ref, slippage_bps)
            per_lot_cost = conservative_price * float(step_size)
            fee_buffer = self._fee_buffer(min_notional, portfolio_state)
            if cash < per_lot_cost + fee_buffer or cash < min_notional:
                return []
            if per_lot_cost <= 0:
                return []
            max_affordable_lots = int(math.floor((cash - fee_buffer) / per_lot_cost))
            desired_lots = min(desired_lots, current_lots + max_affordable_lots)
        delta_lots = desired_lots - current_lots
        if delta_lots == 0:
            return []

        side = OrderSide.BUY if delta_lots > 0 else OrderSide.SELL
        total_lots = abs(delta_lots)
        total_qty = float(self._qty_from_lots(total_lots, step_size))
        total_notional = total_qty * price_ref
        if total_qty < min_qty or total_notional < min_notional:
            return []
        slices = max(1, int(self.slices))
        lots_base = total_lots // slices
        lots_remainder = total_lots % slices
        log_debug(
            self._logger,
            "TWAPPolicy computed slice parameters",
            side=side,
            lots_total=total_lots,
            lots_each=lots_base,
            lots_remainder=lots_remainder,
            step_size=str(step_size),
            price_ref=price_ref,
            equity=equity,
            current_position_qty=current_position_qty,
            desired_qty=desired_qty,
        )

        orders = []
        for i in range(slices):
            lots = lots_base + (1 if i < lots_remainder else 0)
            if lots <= 0:
                continue
            qty_each = float(self._qty_from_lots(lots, step_size))
            log_debug(
                self._logger,
                "TWAPPolicy generated slice order",
                index=i,
                side=side,
                qty=qty_each,
                lots=lots,
                step_size=str(step_size),
            )
            orders.append(
                Order(
                    symbol=self.symbol,
                    side=side,
                    qty=qty_each,
                    order_type=OrderType.MARKET,
                    price=None,
                    tag=f"twap_{i}"
                )
            )
        return orders
