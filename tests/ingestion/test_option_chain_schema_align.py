from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

import ingestion.option_chain.source as option_chain_source

def test_option_chain_raw_align_allows_merged_fields() -> None:
    df = pd.DataFrame(
        [
            {
                "instrument_name": "BTC-1JAN24-10000-C",
                "expiration_timestamp": 1_700_100_000_000,
                "bid_price": 1.0,
                "underlying_price": 30000.0,
                "market_ts": 1_700_000_000_000,
                "arrival_ts": 1_700_000_000_100,
                "aux": {"foo": "bar"},
            }
        ]
    )
    aligned = option_chain_source._align_raw(
        df,
        allowed_cols=option_chain_source._RAW_OPTION_CHAIN_COLUMNS,
        path=Path("dummy.parquet"),
    )
    for col in ("instrument_name", "bid_price", "underlying_price", "market_ts", "arrival_ts", "aux"):
        assert col in aligned.columns
    assert "ask_price" in aligned.columns


def test_option_chain_raw_align_packs_unknown_columns_to_aux() -> None:
    df = pd.DataFrame(
        [
            {
                "instrument_name": "BTC-1JAN24-10000-C",
                "expiration_timestamp": 1_700_100_000_000,
                "totally_unknown": 123,
            }
        ]
    )
    aligned = option_chain_source._align_raw(
        df,
        allowed_cols=option_chain_source._RAW_OPTION_CHAIN_COLUMNS,
        path=Path("dummy.parquet"),
    )
    assert "totally_unknown" not in aligned.columns
    assert "aux" in aligned.columns
    aux = cast(dict[str, Any], aligned.loc[0, "aux"])
    assert aux["totally_unknown"] == 123
