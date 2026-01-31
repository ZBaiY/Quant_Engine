from __future__ import annotations

import pandas as pd

from ingestion.contracts.tick import IngestionTick
from quant_engine.data.derivatives.option_chain.chain_handler import OptionChainDataHandler
from quant_engine.data.derivatives.option_chain.snapshot import OptionChainSnapshot


def test_option_chain_snapshot_splits_frames_and_chain_df_merges() -> None:
    data_ts = 1_700_000_000_000
    df = pd.DataFrame(
        [
            {
                "instrument_name": "BTC-1JAN24-10000-C",
                "expiration_timestamp": 1_700_100_000_000,
                "strike": 10_000,
                "option_type": "call",
                "state": "open",
                "is_active": True,
                "instrument_id": 123,
                "settlement_currency": "BTC",
                "base_currency": "BTC",
                "quote_currency": "USD",
                "contract_size": 1,
                "tick_size": 0.1,
                "min_trade_amount": 0.1,
                "kind": "option",
                "instrument_type": "option",
                "price_index": "btc_usd",
                "counter_currency": "USD",
                "settlement_period": "perpetual",
                "tick_size_steps": 1,
                "bid_price": 1.0,
                "ask_price": 1.1,
                "mid_price": 1.05,
                "last": 1.02,
                "mark_price": 1.03,
                "open_interest": 100,
                "volume_24h": 10.0,
                "volume_usd_24h": 1000.0,
                "mark_iv": 0.5,
                "high": 1.2,
                "low": 0.9,
                "price_change": 0.01,
                "underlying_price": 30000.0,
                "underlying_index": "btc_usd",
                "estimated_delivery_price": 29950.0,
                "interest_rate": 0.01,
                "maker_commission": 0.0002,
                "taker_commission": 0.0005,
                "creation_timestamp": 1_700_000_000_123,
                "aux_chain_data_ts": data_ts,
                "aux_chain_arrival_ts": data_ts + 10,
                "market_ts": 1_700_000_000_111,
                "row_data_ts": data_ts,
                "fetch_step_ts": data_ts,
                "arrival_ts": data_ts + 20,
            }
        ]
    )

    snap = OptionChainSnapshot.from_chain_aligned(
        data_ts=data_ts,
        symbol="BTC",
        chain=df,
        drop_aux=False,
        schema_version=3,
    )

    chain_expected = {
        "instrument_name",
        "expiration_timestamp",
        "expiry_ts",
        "strike",
        "option_type",
        "cp",
        "state",
        "is_active",
        "instrument_id",
        "settlement_currency",
        "base_currency",
        "quote_currency",
        "contract_size",
        "tick_size",
        "min_trade_amount",
        "kind",
        "instrument_type",
        "price_index",
        "counter_currency",
        "settlement_period",
        "tick_size_steps",
    }
    quote_expected = {
        "instrument_name",
        "bid_price",
        "ask_price",
        "mid_price",
        "last",
        "mark_price",
        "open_interest",
        "volume_24h",
        "volume_usd_24h",
        "mark_iv",
        "high",
        "low",
        "market_ts",
        "price_change",
    }
    underlying_expected = {
        "instrument_name",
        "underlying_price",
        "underlying_index",
        "estimated_delivery_price",
        "interest_rate",
    }

    assert set(snap.chain_frame.columns) == chain_expected
    assert set(snap.quote_frame.columns) == quote_expected
    assert set(snap.underlying_frame.columns) == underlying_expected
    assert "maker_commission" in snap.aux_frame.columns
    assert "row_data_ts" in snap.aux_frame.columns
    assert "fetch_step_ts" in snap.aux_frame.columns
    assert "arrival_ts" in snap.aux_frame.columns

    handler = OptionChainDataHandler(symbol="BTC", interval="1m", preset="option_chain")
    tick = IngestionTick(
        timestamp=data_ts,
        data_ts=data_ts,
        domain="option_chain",
        symbol="BTC",
        payload={"data_ts": data_ts, "records": df.to_dict(orient="records")},
        source_id=getattr(handler, "source_id", None),
    )
    handler.on_new_tick(tick)

    out = handler.chain_df(columns=["instrument_name", "bid_price", "underlying_price", "data_ts"])
    assert out.loc[0, "instrument_name"] == "BTC-1JAN24-10000-C"
    assert out.loc[0, "bid_price"] == 1.0
    assert out.loc[0, "underlying_price"] == 30000.0
    assert out.loc[0, "data_ts"] == data_ts
