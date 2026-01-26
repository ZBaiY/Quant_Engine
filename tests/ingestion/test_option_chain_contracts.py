from __future__ import annotations

import copy

import pandas as pd

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
