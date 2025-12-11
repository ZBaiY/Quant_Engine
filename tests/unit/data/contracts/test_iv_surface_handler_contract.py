import pytest
from dataclasses import dataclass
from typing import Dict, Any, Optional

from quant_engine.data.derivatives.iv.snapshot import IVSurfaceSnapshot
from quant_engine.data.derivatives.option_chain.snapshot import OptionChainSnapshot
from quant_engine.data.derivatives.iv.iv_handler import IVSurfaceDataHandler


# ------------------------------------------------------------
from quant_engine.data.derivatives.option_chain.chain_handler import OptionChainDataHandler

# Dummy OptionChainDataHandler with correct semantics
# ------------------------------------------------------------
class DummyChainHandler(OptionChainDataHandler):
    def __init__(self):
        self.snapshots = {}  # ts -> OptionChainSnapshot

    def inject(self, ts, chain):
        snap = OptionChainSnapshot.from_chain(timestamp=ts, chain=chain)
        self.snapshots[ts] = snap

    def get_snapshot(self, ts):
        # emulate anti-lookahead ≤ ts
        eligible = [snap for t, snap in self.snapshots.items() if t <= ts]
        if not eligible:
            return None
        return sorted(eligible, key=lambda s: s.timestamp)[-1]


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------
def test_on_tick_constructs_iv_surface_correctly():
    chain_handler = DummyChainHandler()

    chain_handler.inject(
        100.0,
        [
            {"strike": 30000, "iv": 0.21, "type": "call", "moneyness": 0.01},
            {"strike": 32000, "iv": 0.23, "type": "put",  "moneyness": -0.02},
        ]
    )

    h = IVSurfaceDataHandler(symbol="BTCUSDT", chain_handler=chain_handler)

    snap = h.on_tick(100.0)
    assert snap is not None

    # ATM IV should match closest moneyness → 0.21 (strike 30000)
    assert snap.atm_iv == pytest.approx(0.21)

    # skew = call - put = 0.21 - 0.23 = -0.02
    assert snap.skew == pytest.approx(-0.02)

    # smile extracted from chain
    assert snap.curve == {"30000": 0.21, "32000": 0.23}


def test_get_snapshot_alignment():
    chain_handler = DummyChainHandler()

    chain_handler.inject(90, [{"strike": 30000, "iv": 0.20, "type": "call", "moneyness": 0.01}])
    chain_handler.inject(100, [{"strike": 30000, "iv": 0.25, "type": "call", "moneyness": 0.01}])

    h = IVSurfaceDataHandler(symbol="BTC", chain_handler=chain_handler)
    h.on_tick(90)
    h.on_tick(100)

    snap = h.get_snapshot(95)
    assert snap is not None
    assert snap.atm_iv == 0.20  # cannot look ahead

    snap2 = h.get_snapshot(150)
    assert snap2 is not None
    assert snap2.atm_iv == 0.25


def test_window_returns_proper_sequence():
    chain_handler = DummyChainHandler()

    for ts in [100, 200, 300]:
        chain_handler.inject(ts, [{"strike": 30000, "iv": 0.2 + ts / 1000, "type": "call"}])

    h = IVSurfaceDataHandler(symbol="BTC", chain_handler=chain_handler)

    h.on_tick(100)
    h.on_tick(200)
    h.on_tick(300)

    w = h.window(250, n=5)
    assert len(w) == 2
    assert [s.timestamp for s in w] == [100, 200]


def test_latency_is_correct():
    chain_handler = DummyChainHandler()

    chain_handler.inject(90, [{"strike": 30000, "iv": 0.2, "type": "call"}])

    h = IVSurfaceDataHandler(symbol="BTC", chain_handler=chain_handler)

    snap = h.on_tick(100)
    assert snap is not None
    assert snap.latency == pytest.approx(100 - 90)


def test_empty_handler_behaviour():
    chain_handler = DummyChainHandler()
    h = IVSurfaceDataHandler(symbol="BTC", chain_handler=chain_handler)

    assert h.get_snapshot(10) is None
    assert h.window(10, 3) == []