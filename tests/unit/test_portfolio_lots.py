from quant_engine.data.ohlcv.snapshot import OHLCVSnapshot
from quant_engine.execution.policy.immediate import ImmediatePolicy
from quant_engine.portfolio.fractional import FractionalPortfolioManager
from quant_engine.portfolio.manager import PortfolioManager, EPS


def _make_market_data(price: float) -> dict:
    snap = OHLCVSnapshot.from_bar_aligned(
        timestamp=0,
        bar={
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0.0,
            "data_ts": 0,
        },
        symbol="BTCUSDT",
    )
    return {"ohlcv": snap}


def test_fractional_step_size_floor():
    pm = FractionalPortfolioManager(
        symbol="BTCUSDT",
        initial_capital=1000.0,
        min_qty=0.0,
        min_notional=0.0,
        step_size="0.0001",
    )
    pm.apply_fill({"symbol": "BTCUSDT", "fill_price": 100.0, "filled_qty": 0.00015, "fee": 0.0, "side": "BUY"})
    state = pm.state().to_dict()

    assert state["position_lots"] == 1
    assert abs(state["position_qty"] - 0.0001) < EPS
    assert state["position_qty_str"] == "0.0001"


def test_no_micro_orders_when_delta_lots_zero():
    pm = FractionalPortfolioManager(
        symbol="BTCUSDT",
        initial_capital=1000.0,
        min_qty=0.0,
        min_notional=0.0,
        step_size="0.1",
    )
    pm.apply_fill({"symbol": "BTCUSDT", "fill_price": 100.0, "filled_qty": 0.3, "fee": 0.0, "side": "BUY"})
    state = pm.state().to_dict()

    policy = ImmediatePolicy(symbol="BTCUSDT")
    market_data = _make_market_data(100.0)
    target_position = 0.03  # desired_qty=0.3 at equity=1000

    orders = policy.generate(target_position, state, market_data)
    assert orders == []


def test_cash_constraint_lots_reclip():
    pm = FractionalPortfolioManager(
        symbol="BTCUSDT",
        initial_capital=10.0,
        min_qty=0.0,
        min_notional=0.0,
        step_size="0.1",
    )
    pm.apply_fill({"symbol": "BTCUSDT", "fill_price": 9.0, "filled_qty": 2.0, "fee": 0.0, "side": "BUY"})
    state = pm.state().to_dict()

    assert abs(state["position_qty"] - 1.1) < EPS
    assert pm.cash >= 0.0


def test_standard_step_size_one():
    pm = PortfolioManager(symbol="BTCUSDT", initial_capital=1000.0, min_qty=1, min_notional=1.0, step_size="1")
    pm.apply_fill({"symbol": "BTCUSDT", "fill_price": 100.0, "filled_qty": 1.9, "fee": 0.0, "side": "BUY"})
    state = pm.state().to_dict()

    assert state["position_lots"] == 1
    assert abs(state["position_qty"] - 1.0) < EPS
