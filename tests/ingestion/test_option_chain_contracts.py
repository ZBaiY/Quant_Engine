from __future__ import annotations

import copy

import pandas as pd
import pytest

from ingestion.contracts.tick import IngestionTick
from ingestion.option_chain.normalize import DeribitOptionChainNormalizer


def test_option_chain_normalizer_coerces_data_ts_and_preserves_input() -> None:
    raw = {
        "data_ts": 1_700_000_000,  # seconds
        "arrival_ts": 1_700_000_000,  # legacy alias for arrival authority
        "records": [
            {
                "instrument_name": "BTC-1JAN24-10000-C",
                "expiration_timestamp": 1_700_100_000_000,
                "strike": "10000",
                "option_type": "call",
            }
        ],
    }
    raw_copy = copy.deepcopy(raw)

    normalizer = DeribitOptionChainNormalizer(symbol="BTC")
    tick = normalizer.normalize(raw=raw)

    assert raw == raw_copy
    assert isinstance(tick, IngestionTick)
    assert isinstance(tick.data_ts, int)
    assert tick.data_ts == 1_700_000_000_000
    assert tick.domain == "option_chain"
    assert tick.symbol == "BTC"
    assert isinstance(tick.payload["frame"], pd.DataFrame)


def test_option_chain_normalizer_accepts_frame_payload() -> None:
    data_ts = 1_700_000_000_000
    df = pd.DataFrame(
        [
            {
                "instrument_name": "BTC-1JAN24-10000-C",
                "expiration_timestamp": 1_700_100_000_000,
                "strike": 10_000,
                "option_type": "call",
            }
        ]
    )
    raw = {"data_ts": data_ts, "frame": df}

    normalizer = DeribitOptionChainNormalizer(symbol="BTC")
    tick = normalizer.normalize(raw=raw)

    assert isinstance(tick.payload["frame"], pd.DataFrame)
    assert not tick.payload["frame"].empty


def test_option_chain_normalizer_accepts_empty_records() -> None:
    data_ts = 1_700_000_000_000
    raw = {"data_ts": data_ts, "records": []}

    normalizer = DeribitOptionChainNormalizer(symbol="BTC")
    tick = normalizer.normalize(raw=raw)

    assert isinstance(tick.payload["frame"], pd.DataFrame)
    assert tick.payload["frame"].empty


def test_option_chain_normalizer_rejects_missing_payload() -> None:
    data_ts = 1_700_000_000_000
    raw = {"data_ts": data_ts}

    normalizer = DeribitOptionChainNormalizer(symbol="BTC")
    with pytest.raises(ValueError):
        normalizer.normalize(raw=raw)
