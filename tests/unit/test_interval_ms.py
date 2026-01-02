from __future__ import annotations

from ingestion.contracts.tick import _to_interval_ms


def test_to_interval_ms_valid() -> None:
    assert _to_interval_ms("250ms") == 250
    assert _to_interval_ms("1s") == 1000
    assert _to_interval_ms("1m") == 60_000
    assert _to_interval_ms("2h") == 7_200_000


def test_to_interval_ms_invalid() -> None:
    assert _to_interval_ms("") is None
    assert _to_interval_ms("abc") is None
    assert _to_interval_ms("10x") is None
