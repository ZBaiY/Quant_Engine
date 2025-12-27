from __future__ import annotations

import pytest

from ingestion.contracts.tick import _to_interval_ms, IngestionTick

def test_to_interval_ms_basic():
    assert _to_interval_ms("1m") == 60_000
    assert _to_interval_ms("15m") == 15 * 60_000
    assert _to_interval_ms("1h") == 60 * 60_000

@pytest.mark.parametrize("bad", ["", "0m", "1x", "m1", "1", None])
def test_to_interval_ms_rejects_bad(bad):
    with pytest.raises(Exception):
        _to_interval_ms(bad)  # type: ignore[arg-type]

def test_ingestion_tick_has_required_fields():
    t = IngestionTick(timestamp=1700000000000, data_ts=1700000000123, symbol="BTCUSDT", source="test", payload={})
    assert t.timestamp == 1700000000000
    assert t.data_ts == 1700000000123
    assert t.symbol == "BTCUSDT"
