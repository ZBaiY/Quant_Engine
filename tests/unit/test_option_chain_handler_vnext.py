from __future__ import annotations

from typing import cast

import pandas as pd

from ingestion.contracts.tick import IngestionTick
from quant_engine.data.derivatives.option_chain.chain_handler import OptionChainDataHandler
from quant_engine.strategy.base import GLOBAL_PRESETS


def _make_frame(data_ts: int, market_ts: int, expiry_ts: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instrument_name": "BTC-1-EXP-C",
                "expiration_timestamp": expiry_ts,
                "strike": 10_000,
                "option_type": "call",
                "bid_price": 1.0,
                "ask_price": 1.1,
                "market_ts": market_ts,
                "underlying_price": 30_000.0,
            },
            {
                "instrument_name": "BTC-1-EXP-P",
                "expiration_timestamp": expiry_ts,
                "strike": 10_000,
                "option_type": "put",
                "bid_price": 0.9,
                "ask_price": 1.0,
                "market_ts": market_ts,
                "underlying_price": 30_000.0,
            },
        ]
    )


def test_option_chain_timestamp_semantics_coords_and_selection() -> None:
    data_ts = 1_700_000_000_000
    market_ts = data_ts - 5_000
    expiry_ts = data_ts + 10_000
    handler = OptionChainDataHandler(symbol="BTC", interval="1m", preset="option_chain")
    frame = _make_frame(data_ts, market_ts, expiry_ts)
    tick = IngestionTick(
        timestamp=data_ts,
        data_ts=data_ts,
        domain="option_chain",
        symbol="BTC",
        payload={"data_ts": data_ts, "frame": frame},
        source_id=getattr(handler, "source_id", None),
    )
    handler.on_new_tick(tick)

    coords_df, meta = handler.coords_frame()
    assert int(meta["snapshot_data_ts"]) == data_ts
    assert int(meta["snapshot_market_ts"]) == market_ts
    assert coords_df["snapshot_data_ts"].iloc[0] == data_ts
    assert coords_df["snapshot_market_ts"].iloc[0] == market_ts

    tau_target = expiry_ts - market_ts
    tau_df, tau_meta = handler.select_tau(tau_ms=int(tau_target))
    assert int(tau_meta["snapshot_data_ts"]) == data_ts
    assert int(tau_meta["snapshot_market_ts"]) == market_ts
    assert not tau_df.empty


def test_option_chain_empty_payload_skips_without_throw() -> None:
    data_ts = 1_700_000_000_000
    handler = OptionChainDataHandler(symbol="BTC", interval="1m", preset="option_chain", quality_mode="TRADING")
    tick = IngestionTick(
        timestamp=data_ts,
        data_ts=data_ts,
        domain="option_chain",
        symbol="BTC",
        payload={"data_ts": data_ts, "frame": pd.DataFrame()},
        source_id=getattr(handler, "source_id", None),
    )
    handler.on_new_tick(tick)
    assert handler.last_timestamp() is None


def test_option_chain_selection_avoids_dataframe_truthiness() -> None:
    data_ts = 1_700_000_000_000
    handler = OptionChainDataHandler(symbol="BTC", interval="1m", preset="option_chain", quality_mode="STRICT")
    tick = IngestionTick(
        timestamp=data_ts,
        data_ts=data_ts,
        domain="option_chain",
        symbol="BTC",
        payload={"data_ts": data_ts, "frame": pd.DataFrame()},
        source_id=getattr(handler, "source_id", None),
    )
    handler.on_new_tick(tick)
    df, meta = handler.select_tau(tau_ms=10_000)
    assert df.empty
    assert meta["state"] in {"HARD_FAIL", "SOFT_DEGRADED", "OK"}


def test_coords_frame_cached_once_per_snapshot(monkeypatch) -> None:
    data_ts = 1_700_000_000_000
    market_ts = data_ts - 5_000
    expiry_ts = data_ts + 10_000
    handler = OptionChainDataHandler(symbol="BTC", interval="1m", preset="option_chain")
    frame = _make_frame(data_ts, market_ts, expiry_ts)
    tick = IngestionTick(
        timestamp=data_ts,
        data_ts=data_ts,
        domain="option_chain",
        symbol="BTC",
        payload={"data_ts": data_ts, "frame": frame},
        source_id=getattr(handler, "source_id", None),
    )
    handler.on_new_tick(tick)

    calls = {"n": 0}
    original = handler._coords_frame_uncached

    def _wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(handler, "_coords_frame_uncached", _wrapped)
    handler.coords_frame()
    handler.coords_frame()
    assert calls["n"] == 1
    handler.coords_frame(x_axis="moneyness")
    assert calls["n"] == 2


def test_select_point_cp_policy_same_prefers_same_cp_linear_x() -> None:
    data_ts = 1_700_000_000_000
    market_ts = data_ts - 5_000
    expiry_ts = data_ts + 86_400_000
    handler = OptionChainDataHandler(symbol="BTC", interval="1m", preset="option_chain")
    frame = pd.DataFrame(
        [
            {
                "instrument_name": "BTC-EXP-105-C",
                "expiration_timestamp": expiry_ts,
                "strike": 105,
                "cp": "C",
                "mark_price": 10.0,
                "market_ts": market_ts,
                "underlying_price": 100.0,
            },
            {
                "instrument_name": "BTC-EXP-115-C",
                "expiration_timestamp": expiry_ts,
                "strike": 115,
                "cp": "C",
                "mark_price": 12.0,
                "market_ts": market_ts,
                "underlying_price": 100.0,
            },
            {
                "instrument_name": "BTC-EXP-85-P",
                "expiration_timestamp": expiry_ts,
                "strike": 85,
                "cp": "P",
                "mark_price": 1.0,
                "market_ts": market_ts,
                "underlying_price": 100.0,
            },
            {
                "instrument_name": "BTC-EXP-95-P",
                "expiration_timestamp": expiry_ts,
                "strike": 95,
                "cp": "P",
                "mark_price": 2.0,
                "market_ts": market_ts,
                "underlying_price": 100.0,
            },
        ]
    )
    tick = IngestionTick(
        timestamp=data_ts,
        data_ts=data_ts,
        domain="option_chain",
        symbol="BTC",
        payload={"data_ts": data_ts, "frame": frame},
        source_id=getattr(handler, "source_id", None),
    )
    handler.on_new_tick(tick)

    target_tau = expiry_ts - market_ts
    point, meta = handler.select_point(
        tau_ms=int(target_tau),
        x=0.02,
        x_axis="log_moneyness",
        interp="linear_x",
        cp_policy="same",
        price_field="mark_price",
    )
    assert point is not None
    value = point["value_fields"]["mark_price"]
    assert value > 5.0
    assert meta["state"] in {"OK", "SOFT_DEGRADED", "HARD_FAIL"}


def test_window_for_tau_selection_weight_propagates() -> None:
    data_ts = 1_700_000_000_000
    market_ts = data_ts - 5_000
    expiry_a = data_ts + 20_000
    expiry_b = data_ts + 40_000
    handler = OptionChainDataHandler(symbol="BTC", interval="1m", preset="option_chain")
    frame = pd.DataFrame(
        [
            {
                "instrument_name": "BTC-A",
                "expiration_timestamp": expiry_a,
                "strike": 10_000,
                "option_type": "call",
                "mark_price": 1.0,
                "market_ts": market_ts,
                "underlying_price": 30_000.0,
            },
            {
                "instrument_name": "BTC-B",
                "expiration_timestamp": expiry_b,
                "strike": 10_000,
                "option_type": "call",
                "mark_price": 2.0,
                "market_ts": market_ts,
                "underlying_price": 30_000.0,
            },
        ]
    )
    tick = IngestionTick(
        timestamp=data_ts,
        data_ts=data_ts,
        domain="option_chain",
        symbol="BTC",
        payload={"data_ts": data_ts, "frame": frame},
        source_id=getattr(handler, "source_id", None),
    )
    handler.on_new_tick(tick)

    target_tau = int((expiry_a - market_ts + expiry_b - market_ts) / 2)
    _, meta = handler.select_tau(tau_ms=target_tau, method="bracket")
    views = handler.window_for_tau(
        ts_start=data_ts,
        ts_end=data_ts,
        tau_ms=target_tau,
        step_ms=1,
        method="bracket",
    )
    assert len(views) == 1
    view = views[0]
    frame_out = cast(pd.DataFrame, view.frame)
    assert "selection_weight" in frame_out.columns
    weights = meta["selection"]["weights"]
    selected = meta["selection"]["selected_expiries"]
    weight_map = {int(ex): float(w) for ex, w in zip(selected, weights)}
    for ex in frame_out["expiry_ts"].unique():
        series = cast(pd.Series, frame_out.loc[frame_out["expiry_ts"] == ex, "selection_weight"])
        assert series.iloc[0] == weight_map[int(ex)]


def test_missing_market_ts_sets_reason_and_tau_anchor() -> None:
    data_ts = 1_700_000_000_000
    expiry_ts = data_ts + 10_000
    handler = OptionChainDataHandler(symbol="BTC", interval="1m", preset="option_chain")
    frame = pd.DataFrame(
        [
            {
                "instrument_name": "BTC-EXP-1",
                "expiration_timestamp": expiry_ts,
                "strike": 10_000,
                "option_type": "call",
                "mark_price": 1.0,
                "underlying_price": 30_000.0,
            }
        ]
    )
    tick = IngestionTick(
        timestamp=data_ts,
        data_ts=data_ts,
        domain="option_chain",
        symbol="BTC",
        payload={"data_ts": data_ts, "frame": frame},
        source_id=getattr(handler, "source_id", None),
    )
    handler.on_new_tick(tick)

    _, meta = handler.coords_frame(tau_def="market_ts")
    reason_codes = {r["reason_code"] for r in meta["reasons"]}
    assert "MISSING_MARKET_TS" in reason_codes
    assert meta["tau_anchor_ts"] == data_ts
    assert meta["market_ts_ref_method"] == "missing"


def test_quality_spread_max_from_global_preset_changes_reason() -> None:
    original = GLOBAL_PRESETS["option_chain"]["quality"]["spread_max"]
    try:
        GLOBAL_PRESETS["option_chain"]["quality"]["spread_max"] = 0.01
        data_ts = 1_700_000_000_000
        market_ts = data_ts - 5_000
        expiry_ts = data_ts + 10_000
        handler = OptionChainDataHandler(symbol="BTC", interval="1m", preset="option_chain")
        frame = pd.DataFrame(
            [
                {
                    "instrument_name": "BTC-EXP-1",
                    "expiration_timestamp": expiry_ts,
                    "strike": 10_000,
                    "option_type": "call",
                    "bid_price": 1.0,
                    "ask_price": 1.2,
                    "market_ts": market_ts,
                    "underlying_price": 30_000.0,
                }
            ]
        )
        tick = IngestionTick(
            timestamp=data_ts,
            data_ts=data_ts,
            domain="option_chain",
            symbol="BTC",
            payload={"data_ts": data_ts, "frame": frame},
            source_id=getattr(handler, "source_id", None),
        )
        handler.on_new_tick(tick)
        _, meta = handler.coords_frame()
        reason_codes = {r["reason_code"] for r in meta["reasons"]}
        assert "WIDE_SPREAD" in reason_codes

        GLOBAL_PRESETS["option_chain"]["quality"]["spread_max"] = 1.0
        handler_2 = OptionChainDataHandler(symbol="BTC", interval="1m", preset="option_chain")
        handler_2.on_new_tick(tick)
        _, meta_2 = handler_2.coords_frame()
        reason_codes_2 = {r["reason_code"] for r in meta_2["reasons"]}
        assert "WIDE_SPREAD" not in reason_codes_2
    finally:
        GLOBAL_PRESETS["option_chain"]["quality"]["spread_max"] = original


def test_term_bucket_ms_uses_global_preset_value() -> None:
    original_term = GLOBAL_PRESETS["option_chain"]["term_bucket_ms"]
    original_cache_term = GLOBAL_PRESETS["option_chain"]["cache"]["term_bucket_ms"]
    try:
        GLOBAL_PRESETS["option_chain"]["term_bucket_ms"] = 12_345
        GLOBAL_PRESETS["option_chain"]["cache"]["term_bucket_ms"] = 12_345
        handler = OptionChainDataHandler(symbol="BTC", interval="1m", preset="option_chain")
        assert handler.term_bucket_ms == 12_345
        _, meta = handler.select_tau(tau_ms=10_000)
        assert meta["selection_context"]["term_bucket_ms"] == 12_345
    finally:
        GLOBAL_PRESETS["option_chain"]["term_bucket_ms"] = original_term
        GLOBAL_PRESETS["option_chain"]["cache"]["term_bucket_ms"] = original_cache_term
