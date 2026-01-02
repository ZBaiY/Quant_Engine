from __future__ import annotations

import pytest

from quant_engine.exceptions.core import FatalError
from quant_engine.utils.guards import ensure_epoch_ms, assert_monotonic, assert_no_lookahead


def test_ensure_epoch_ms_seconds_to_ms() -> None:
    assert ensure_epoch_ms(1_622_505_600) == 1_622_505_600_000
    assert ensure_epoch_ms(1_622_505_600.5) == 1_622_505_600_500


def test_ensure_epoch_ms_idempotent_for_ms() -> None:
    assert ensure_epoch_ms(1_622_505_600_000) == 1_622_505_600_000


def test_ensure_epoch_ms_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        ensure_epoch_ms(None)
    with pytest.raises(ValueError):
        ensure_epoch_ms(True)


def test_assert_monotonic_allows_equal() -> None:
    assert assert_monotonic(1000, 1000, label="t") == 1000


def test_assert_monotonic_rejects_backwards() -> None:
    with pytest.raises(FatalError):
        assert_monotonic(999, 1000, label="t")


def test_assert_no_lookahead_allows_equal() -> None:
    assert_no_lookahead(1000, 1000, label="t")


def test_assert_no_lookahead_rejects_future() -> None:
    with pytest.raises(FatalError):
        assert_no_lookahead(1000, 1001, label="t")
