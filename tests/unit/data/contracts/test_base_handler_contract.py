import pytest
import pandas as pd
from typing import List
from dataclasses import dataclass

# ------------------------------------------
# DUMMY IMPLEMENTATION FOR CONTRACT TESTING
# ------------------------------------------

@dataclass
class DummyUnit:
    ts: float
    value: float


class DummyHandler:
    """
    This dummy handler is ONLY used to test the BaseDataHandler contract.
    It simulates a deterministic time-aligned sequence of data.
    """

    def __init__(self):
        # internal buffer sorted by timestamp
        self.data: List[DummyUnit] = []

    # --- metadata ---
    @property
    def symbol(self) -> str:
        return "TEST_SYMBOL"

    # --- required API ---
    def latest(self):
        if not self.data:
            return None
        return self.data[-1]

    def latest_timestamp(self):
        if not self.data:
            return None
        return self.data[-1].ts

    def get_snapshot(self, ts: float):
        """
        Must return last item with timestamp ≤ ts.
        Must not look ahead.
        """
        eligible = [u for u in self.data if u.ts <= ts]
        if not eligible:
            return None
        return eligible[-1]

    def window(self, ts: float, n: int):
        """
        Must return the last n entries with timestamp ≤ ts.
        Anti-lookahead is required.
        """
        eligible = [u for u in self.data if u.ts <= ts]
        return eligible[-n:]

    def ready(self) -> bool:
        return len(self.data) > 0

    def flush_cache(self):
        self.data.clear()


# ==========================================
# CONTRACT TESTS
# ==========================================

def test_contract_symbol_exists():
    handler = DummyHandler()
    assert isinstance(handler.symbol, str)


def test_contract_latest_returns_last_item():
    handler = DummyHandler()
    handler.data = [DummyUnit(1.0, 10), DummyUnit(2.0, 20)]
    latest = handler.latest()

    assert isinstance(latest, DummyUnit)
    assert latest.ts == 2.0
    assert latest.value == 20


def test_contract_latest_timestamp():
    handler = DummyHandler()
    handler.data = [DummyUnit(1.0, 10), DummyUnit(2.0, 20)]

    ts = handler.latest_timestamp()
    assert ts == 2.0
    assert isinstance(ts, float)


def test_contract_ready_flag():
    handler = DummyHandler()
    assert handler.ready() is False

    handler.data.append(DummyUnit(1.0, 10))
    assert handler.ready() is True


def test_contract_snapshot_alignment_and_no_lookahead():
    handler = DummyHandler()
    handler.data = [
        DummyUnit(1.0, 10),
        DummyUnit(2.0, 20),
        DummyUnit(3.0, 30),
    ]

    snap = handler.get_snapshot(2.5)
    assert isinstance(snap, DummyUnit)
    assert snap.ts == 2.0  # cannot look ahead to 3.0
    assert snap.value == 20


def test_contract_snapshot_none_if_no_past_data():
    handler = DummyHandler()
    handler.data = [
        DummyUnit(5.0, 99),
    ]
    snap = handler.get_snapshot(1.0)
    assert snap is None  # nothing ≤ 1.0


def test_contract_window_alignment_and_no_lookahead():
    handler = DummyHandler()
    handler.data = [
        DummyUnit(1.0, 10),
        DummyUnit(2.0, 20),
        DummyUnit(3.0, 30),
        DummyUnit(4.0, 40),
    ]

    w = handler.window(ts=3.5, n=2)
    assert len(w) == 2
    assert [u.ts for u in w] == [2.0, 3.0]  # ≤ ts only, no lookahead


def test_contract_window_respects_n_too_big():
    handler = DummyHandler()
    handler.data = [
        DummyUnit(1.0, 10),
        DummyUnit(2.0, 20),
    ]

    w = handler.window(ts=10.0, n=5)
    assert len(w) == 2  # available data only


def test_contract_flush_cache():
    handler = DummyHandler()
    handler.data = [DummyUnit(1.0, 10)]
    handler.flush_cache()
    assert handler.data == []
    assert handler.ready() is False