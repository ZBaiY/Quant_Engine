import pandas as pd

from quant_engine.data.derivatives.option_chain.cache import (
    OptionChainExpiryIndexedCache,
    OptionChainTermBucketedCache,
)
from quant_engine.data.derivatives.option_chain.snapshot import OptionChainSnapshot, OptionChainSnapshotView


def _make_snapshot(data_ts: int, expiries: list[int], symbol: str = "BTC") -> OptionChainSnapshot:
    rows = []
    for i, ex in enumerate(expiries):
        rows.append(
            {
                "instrument_name": f"{symbol}-{i}-{ex}",
                "expiration_timestamp": int(ex),
                "strike": 10_000 + i,
                "option_type": "call",
            }
        )
    df = pd.DataFrame(rows)
    return OptionChainSnapshot.from_chain_aligned(data_ts=data_ts, symbol=symbol, chain=df)


def test_by_expiry_time_correct_out_of_order_push() -> None:
    cache = OptionChainExpiryIndexedCache(maxlen=10)
    expiry = 20_000
    s1 = _make_snapshot(3000, [expiry])
    s2 = _make_snapshot(1000, [expiry])
    s3 = _make_snapshot(2000, [expiry])
    cache.push(s1)
    cache.push(s2)
    cache.push(s3)

    snap = cache.get_at_or_before_for_expiry(expiry, 2500)
    assert snap is not None
    assert snap.data_ts == 2000
    assert set(snap.frame["expiry_ts"].unique()) == {expiry}


def test_by_term_time_correct_out_of_order_push() -> None:
    cache = OptionChainTermBucketedCache(
        maxlen=10,
        term_bucket_ms=5000,
    )
    expiry = 20_000
    term_key = 15_000
    s1 = _make_snapshot(3000, [expiry])
    s2 = _make_snapshot(1000, [expiry])
    s3 = _make_snapshot(2000, [expiry])
    cache.push(s1)
    cache.push(s2)
    cache.push(s3)

    snap = cache.get_at_or_before_for_term(term_key, 2500)
    assert snap is not None
    assert snap.data_ts == 2000
    assert set(snap.frame["expiry_ts"].unique()) == {expiry}


def test_by_term_dedup_same_snapshot_in_bucket() -> None:
    cache = OptionChainTermBucketedCache(
        maxlen=10,
        term_bucket_ms=10_000,
        default_term_window=3,
    )
    snap = _make_snapshot(1000, [15_000, 18_000])
    cache.push(snap)

    term_key = 10_000
    view = cache.get_at_or_before_for_term(term_key, 1000)
    assert view is not None
    expiries = set(view.frame["expiry_ts"].unique())
    assert expiries == {15_000, 18_000}


def test_snapshot_view_frame_tags_and_slice() -> None:
    df = pd.DataFrame(
        [
            {
                "instrument_name": "BTC-0-11111",
                "expiration_timestamp": 11_111,
                "strike": 10_000,
                "option_type": "call",
                "bid_price": 1.0,
                "ask_price": 1.1,
                "market_ts": 1200,
            },
            {
                "instrument_name": "BTC-1-22222",
                "expiration_timestamp": 22_222,
                "strike": 10_100,
                "option_type": "call",
                "bid_price": 1.0,
                "ask_price": 1.1,
                "market_ts": 1250,
            },
        ]
    )
    base = OptionChainSnapshot.from_chain_aligned(data_ts=1234, symbol="BTC", chain=df)
    view = OptionChainSnapshotView.for_expiry(base=base, expiry_ts=11_111)

    frame = view.frame
    assert "snapshot_data_ts" in frame.columns
    assert set(frame["snapshot_data_ts"].unique()) == {1234}
    assert "snapshot_market_ts" in frame.columns
    assert set(frame["snapshot_market_ts"].unique()) == {1225}
    assert set(frame["slice_kind"].unique()) == {"expiry"}
    assert set(frame["slice_key"].unique()) == {11_111}
    assert set(frame["expiry_ts"].unique()) == {11_111}
    assert "snapshot_data_ts" not in base.frame.columns


def test_snapshot_cached_keys_and_missing_expiry_view() -> None:
    base = _make_snapshot(1000, [11_111, 22_222])
    keys = base.get_expiry_keys_ms()
    assert keys == {11_111, 22_222}
    term_keys = base.get_term_keys_ms(10_000)
    assert term_keys == {10_000, 20_000}

    view = OptionChainSnapshotView.for_expiry(base=base, expiry_ts=33_333)
    frame = view.frame
    assert frame.empty
    assert "snapshot_data_ts" not in base.frame.columns


def test_snapshot_expiry_keys_robust_coercion() -> None:
    df = pd.DataFrame(
        {
            "instrument_name": ["A", "B", "C", "D"],
            "expiry_ts": ["11111", 22222.0, None, "bad"],
            "strike": [1, 2, 3, 4],
            "cp": ["C", "P", "C", "P"],
        }
    )
    snap = OptionChainSnapshot.from_chain_aligned(data_ts=1000, symbol="BTC", chain=df)
    assert snap.get_expiry_keys_ms() == {11111, 22222}


def test_by_expiry_overwrite_same_data_ts() -> None:
    cache = OptionChainExpiryIndexedCache(maxlen=10)
    expiry = 20_000
    s1 = _make_snapshot(1000, [expiry])
    s2 = _make_snapshot(1000, [expiry, 30_000])
    cache.push(s1)
    cache.push(s2)

    view = cache.get_at_or_before_for_expiry(expiry, 1000)
    assert view is not None
    expiries = set(view.frame["expiry_ts"].unique())
    assert expiries == {expiry}


def test_by_expiry_eviction_and_get_at_or_before() -> None:
    cache = OptionChainExpiryIndexedCache(maxlen=2)
    expiry = 20_000
    cache.push(_make_snapshot(1000, [expiry]))
    cache.push(_make_snapshot(2000, [expiry]))
    cache.push(_make_snapshot(3000, [expiry]))
    assert cache.get_at_or_before_for_expiry(expiry, 1500) is None
    cache_1 = cache.get_at_or_before_for_expiry(expiry, 2500)
    assert cache_1 is not None
    assert cache_1.data_ts == 2000
    cache_2 = cache.get_at_or_before_for_expiry(expiry, 3500)
    assert cache_2 is not None
    assert cache_2.data_ts == 3000


def test_cache_requires_main_capacity_for_expiry_index() -> None:
    cache = OptionChainExpiryIndexedCache(maxlen=3, default_expiry_window=2)
    expiry = 20_000
    cache.push(_make_snapshot(1000, [expiry]))
    cache.push(_make_snapshot(2000, [expiry]))
    cache.push(_make_snapshot(3000, [expiry]))
    df = cache.window_df_for_expiry(expiry)
    assert set(df["snapshot_data_ts"].unique()) == {2000, 3000}


def test_cache_requires_main_capacity_for_term_index() -> None:
    cache = OptionChainTermBucketedCache(
        maxlen=3,
        term_bucket_ms=5000,
        default_term_window=2,
    )
    expiry = 20_000
    cache.push(_make_snapshot(1000, [expiry]))
    cache.push(_make_snapshot(2000, [expiry]))
    cache.push(_make_snapshot(3000, [expiry]))
    df = cache.window_df_for_term(15_000)
    assert set(df["snapshot_data_ts"].unique()) == {2000, 3000}


def test_cache_ordering_ignores_fetch_step_ts() -> None:
    cache = OptionChainExpiryIndexedCache(maxlen=10)
    df_a = pd.DataFrame(
        [
            {
                "instrument_name": "BTC-0-1000",
                "expiration_timestamp": 20_000,
                "strike": 10_000,
                "option_type": "call",
                "fetch_step_ts": 9_999,
            }
        ]
    )
    df_b = pd.DataFrame(
        [
            {
                "instrument_name": "BTC-1-2000",
                "expiration_timestamp": 20_000,
                "strike": 10_001,
                "option_type": "call",
                "fetch_step_ts": 1,
            }
        ]
    )
    s1 = OptionChainSnapshot.from_chain_aligned(data_ts=1000, symbol="BTC", chain=df_a, drop_aux=False)
    s2 = OptionChainSnapshot.from_chain_aligned(data_ts=2000, symbol="BTC", chain=df_b, drop_aux=False)
    cache.push(s2)
    cache.push(s1)
    snap = cache.get_at_or_before(1500)
    assert snap is not None
    assert snap.data_ts == 1000
