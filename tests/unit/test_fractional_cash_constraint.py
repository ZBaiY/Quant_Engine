from quant_engine.risk.rules_constraints import FractionalCashConstraintRule


class MockSnapshot:
    def __init__(self, close: float, data_ts: int = 0):
        self._close = close
        self.data_ts = data_ts

    def get_attr(self, name):
        if name == "close":
            return self._close
        return None


def test_fractional_cash_constraint_lot_clamp():
    rule = FractionalCashConstraintRule(
        symbol="BTCUSDT",
        fee_rate=0.001,
        slippage_bound_bps=10.0,
        min_notional=10.0,
    )
    mark = 90_699.8255
    slip = 10.0 / 1e4
    step = 0.001
    cash = 1_000_000.0
    equity = 1_000_000.0
    per_lot_cost = mark * (1.0 + slip) * (1.0 + 0.001) * step
    max_add_lots = int(cash // per_lot_cost)
    reachable_qty = max_add_lots * step
    reachable_position = (reachable_qty * mark) / equity

    context = {
        "timestamp": 1,
        "portfolio": {
            "cash": cash,
            "total_equity": equity,
            "position_lots": 0,
            "position_qty": 0.0,
            "qty_step": str(step),
            "qty_mode": "LOTS",
            "min_notional": 10.0,
        },
        "primary_snapshots": {"ohlcv": MockSnapshot(mark)},
    }

    adjusted = rule.adjust(1.0, context)

    assert adjusted <= 1.0
    assert abs(adjusted - reachable_position) < 1e-9
