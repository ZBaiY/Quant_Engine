from __future__ import annotations

import threading
from typing import Any, Iterator
import types

import pytest
import requests

import ingestion.option_chain.source as option_chain_source
from ingestion.contracts.tick import IngestionTick
from ingestion.option_chain.source import DeribitOptionChainRESTSource


def _response(payload: Any) -> Any:
    return types.SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: payload,
    )


def test_option_chain_rest_source_monotonic_and_poll_interval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    ts_iter: Iterator[int] = iter([1_000, 2_000])

    def fake_now_ms() -> int:
        return next(ts_iter)

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> Any:
        payload = {
            "result": [
                {
                    "instrument_name": "BTC-1JAN24-10000-C",
                    "expiration_timestamp": 1_700_100_000_000,
                    "strike": 10_000,
                    "option_type": "call",
                }
            ]
        }
        return _response(payload)

    stop_event = threading.Event()
    waits: list[float] = []

    def fake_wait(seconds: float) -> bool:
        waits.append(seconds)
        return len(waits) >= 2

    monkeypatch.setattr(option_chain_source, "_now_ms", fake_now_ms)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(stop_event, "wait", fake_wait)
    monkeypatch.setattr(option_chain_source, "DATA_ROOT", tmp_path)

    src = DeribitOptionChainRESTSource(
        currency="BTC",
        interval="1m",
        poll_interval_ms=500,
        stop_event=stop_event,
        root=tmp_path / "raw" / "option_chain",
    )
    monkeypatch.setattr(src, "_write_raw_snapshot", lambda *_args, **_kwargs: None)

    rows = list(src)

    assert [r["data_ts"] for r in rows] == [1_000, 2_000]
    assert [r["data_ts"] for r in rows] == sorted(r["data_ts"] for r in rows)
    assert waits == [0.5, 0.5]
    assert isinstance(rows[0]["records"], list)
    assert all(not isinstance(r, IngestionTick) for r in rows)
