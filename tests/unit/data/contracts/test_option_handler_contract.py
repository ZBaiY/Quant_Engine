import pytest
from typing import Optional

from quant_engine.data.derivatives.option_chain.option_chain import OptionChain
from quant_engine.data.derivatives.option_chain.snapshot import OptionChainSnapshot
from quant_engine.data.derivatives.option_chain.option_contract import OptionContract, OptionType
from quant_engine.data.derivatives.option_chain.chain_handler import OptionChainDataHandler


# -------------------------------------------------------------------
# Dummy concrete OptionChainDataHandler for contract testing
# -------------------------------------------------------------------
class DummyOptionChainHandler(OptionChainDataHandler):
    """
    A minimal concrete implementation used ONLY for contract tests.
    It emulates:
        • timestamp‑aligned snapshots
        • anti‑lookahead
        • a rolling internal snapshot buffer
    """

    def __init__(self):
        self._snaps = {}   # ts → OptionChainSnapshot
        self._last_ts: Optional[float] = None

    # ---- injection for test only ----
    def inject(self, ts: float, chain_data):
        """
        Test-only injection helper.

        Accepts either:
          - a list of raw dicts (legacy option chain format), or
          - a list of OptionContract objects (preferred contract).

        It always normalizes to a list of plain dicts before calling
        OptionChainSnapshot.from_chain so the downstream contract stays stable.
        """
        normalized = []
        for leg in chain_data:
            if isinstance(leg, OptionContract):
                normalized.append(leg.to_dict())
            else:
                normalized.append(leg)

        snap = OptionChainSnapshot.from_chain(
            timestamp=ts,
            chain=normalized,
        )
        self._snaps[ts] = snap
        self._last_ts = ts

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------
    @property
    def symbol(self) -> str:
        return "TEST"

    def latest(self) -> Optional[OptionChainSnapshot]:
        if not self._snaps:
            return None
        return self._snaps[self._last_ts]

    def get_snapshot(self, ts: float) -> Optional[OptionChainSnapshot]:
        """
        The contract requires timestamp‑aligned, anti‑lookahead snapshots.
        """
        eligible = [s for t, s in self._snaps.items() if t <= ts]
        if not eligible:
            return None
        return sorted(eligible, key=lambda s: s.timestamp)[-1]

    def window(self, ts: float, n: int):
        eligible = [s for t, s in self._snaps.items() if t <= ts]
        eligible_sorted = sorted(eligible, key=lambda s: s.timestamp)
        return eligible_sorted[-n:]

    def ready(self) -> bool:
        return len(self._snaps) > 0

    def latest_timestamp(self) -> Optional[float]:
        return self._last_ts

    def flush_cache(self) -> None:
        self._snaps.clear()
        self._last_ts = None


# -------------------------------------------------------------------
# Contract Tests
# -------------------------------------------------------------------
def test_ready_flag():
    h = DummyOptionChainHandler()
    assert h.ready() is False

    h.inject(100.0, [{"strike": 30000, "iv": 0.2, "type": "call"}])
    assert h.ready() is True


def test_option_contract_mid_and_to_dict():
    opt = OptionContract(
        symbol="BTCUSDT",
        expiry="2025-06-27",
        strike=30000.0,
        option_type=OptionType.CALL,
        bid=10.0,
        ask=12.0,
        last=None,
        volume=100.0,
        open_interest=50.0,
        implied_vol=0.2,
        delta=0.5,
        gamma=0.01,
        vega=0.2,
        theta=-0.01,
    )

    assert opt.mid() == pytest.approx(11.0)

    d = opt.to_dict()
    assert d["symbol"] == "BTCUSDT"
    assert d["option_type"] == "C"
    assert d["strike"] == pytest.approx(30000.0)
    assert d["mid"] == pytest.approx(11.0)


def test_latest_snapshot():
    h = DummyOptionChainHandler()
    h.inject(100.0, [{"strike": 30000, "iv": 0.2, "type": "call"}])

    snap = h.get_snapshot(999.0)
    assert isinstance(snap, OptionChainSnapshot)
    assert snap.atm_iv == pytest.approx(0.2)


def test_option_handler_accepts_option_contract_objects():
    h = DummyOptionChainHandler()

    chain = [
        OptionContract(
            symbol="BTCUSDT",
            expiry="2025-06-27",
            strike=30000.0,
            option_type=OptionType.CALL,
            bid=None,
            ask=None,
            last=None,
            volume=None,
            open_interest=None,
            implied_vol=0.2,
            delta=None,
            gamma=None,
            vega=None,
            theta=None,
        )
    ]

    h.inject(100.0, chain)
    snap = h.get_snapshot(150.0)

    assert snap is not None
    # atm_iv should reflect the implied_vol from the OptionContract
    assert snap.atm_iv == pytest.approx(0.2)


def test_timestamp_alignment_and_no_lookahead():
    h = DummyOptionChainHandler()

    h.inject(90.0, [{"strike": 30000, "iv": 0.18, "type": "call"}])
    h.inject(100.0, [{"strike": 30000, "iv": 0.25, "type": "call"}])

    snap = h.get_snapshot(95.0)
    # Must return the 90 snapshot, not the 100
    assert snap is not None
    assert snap.atm_iv == pytest.approx(0.18)

    snap2 = h.get_snapshot(200.0)
    assert snap2 is not None
    assert snap2.atm_iv == pytest.approx(0.25)


def test_window_behavior():
    h = DummyOptionChainHandler()

    h.inject(100.0, [{"strike": 30000, "iv": 0.20, "type": "call"}])
    h.inject(200.0, [{"strike": 30000, "iv": 0.22, "type": "call"}])
    h.inject(300.0, [{"strike": 30000, "iv": 0.24, "type": "call"}])

    w = h.window(ts=250.0, n=5)
    assert len(w) == 2
    assert [s.timestamp for s in w] == [100.0, 200.0]


def test_smile_and_skew_are_extracted_correctly():
    """
    Ensure OptionChainSnapshot.from_chain() parses chain data fields
    used by downstream handlers (IVSurfaceDataHandler etc.).
    """
    h = DummyOptionChainHandler()

    h.inject(
        100.0,
        [
            {"strike": 30000, "iv": 0.21, "type": "call", "moneyness": 0.01},
            {"strike": 32000, "iv": 0.23, "type": "put",  "moneyness": -0.02},
        ],
    )

    snap = h.get_snapshot(150.0)

    # ATM IV should match closest‑to‑money (strike=30000)
    assert snap is not None
    assert snap.atm_iv == pytest.approx(0.21)

    # skew = call − put = 0.21 − 0.23 = −0.02
    assert snap.skew == pytest.approx(-0.02)

    # smile extracted
    assert snap.smile == {"30000": 0.21, "32000": 0.23}


def test_flush_clears_cache():
    h = DummyOptionChainHandler()
    h.inject(100.0, [{"strike": 30000, "iv": 0.2, "type": "call"}])

    assert h.ready() is True
    h.flush_cache()
    assert h.ready() is False
    assert h.get_snapshot(999.0) is None
