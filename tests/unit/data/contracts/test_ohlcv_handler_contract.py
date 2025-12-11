import pytest
import pandas as pd
from quant_engine.data.ohlcv.cache import DataCache
from quant_engine.data.ohlcv.loader import OHLCVLoader
from quant_engine.data.ohlcv.realtime import RealTimeDataHandler
from quant_engine.data.ohlcv.snapshot import OHLCVSnapshot


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def make_bar(ts, o=1, h=2, l=0.5, c=1.5, v=100):
    return pd.DataFrame([{
        "timestamp": float(ts),
        "open": float(o),
        "high": float(h),
        "low": float(l),
        "close": float(c),
        "volume": float(v),
    }])


# -------------------------------------------------------------------
# DataCache Contract Tests
# -------------------------------------------------------------------
def test_cache_update_and_latest():
    cache = DataCache(window=5)
    bar1 = make_bar(100)
    bar2 = make_bar(200)

    cache.update(bar1)
    cache.update(bar2)

    latest = cache.get_latest()
    assert isinstance(latest, pd.DataFrame)
    assert latest["timestamp"].iloc[0] == 200


def test_cache_window_n():
    cache = DataCache(window=5)
    for ts in [100, 200, 300]:
        cache.update(make_bar(ts))

    df = cache.get_window(2)
    assert len(df) == 2
    assert df["timestamp"].tolist() == [200, 300]


def test_cache_latest_before_ts():
    cache = DataCache()
    cache.update(make_bar(100))
    cache.update(make_bar(200))
    cache.update(make_bar(300))

    snap = cache.latest_before_ts(250)
    assert isinstance(snap, pd.DataFrame)
    assert snap["timestamp"].iloc[0] == 200

    snap2 = cache.latest_before_ts(50)
    assert snap2 is None


def test_cache_window_before_ts():
    cache = DataCache()
    cache.update(make_bar(100))
    cache.update(make_bar(200))
    cache.update(make_bar(300))

    df = cache.window_before_ts(250, 5)
    assert df["timestamp"].tolist() == [100, 200]


# -------------------------------------------------------------------
# OHLCVSnapshot Contract Tests
# -------------------------------------------------------------------
def test_snapshot_from_bar_correct_fields():
    bar = {
        "timestamp": 100,
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.5,
        "volume": 10,
    }
    snap = OHLCVSnapshot.from_bar(120, bar)

    assert snap.timestamp == 100
    assert snap.latency == pytest.approx(20)
    assert snap.open == 1.0
    assert snap.close == 1.5
    assert snap.volume == 10


def test_snapshot_no_lookahead():
    bar = {"timestamp": 200, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}
    snap = OHLCVSnapshot.from_bar(150, bar)  # bar is in the future
    # According to contract, timestamp must remain the bar-ts
    assert snap.timestamp == 200
    assert snap.latency == pytest.approx(-50)  # negative latency allowed


# -------------------------------------------------------------------
# OHLCVLoader Contract Tests
# -------------------------------------------------------------------
def test_loader_standardizes_dataframe():
    df = pd.DataFrame([
        {"timestamp": 100, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
        {"timestamp": 90,  "open": 1.1, "high": 2.1, "low": 0.4, "close": 1.4, "volume": 20},
    ])
    out = OHLCVLoader.from_dataframe(df)

    assert isinstance(out, list)
    assert out[0]["timestamp"] == 90  # sorted
    assert isinstance(out[0]["open"], float)


# -------------------------------------------------------------------
# RealTimeDataHandler Contract Tests
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# RealTimeDataHandler v4 snapshot contract test
# -------------------------------------------------------------------
def test_realtime_handler_snapshot_latest():
    h = RealTimeDataHandler("BTC")
    h.on_new_tick(make_bar(100))
    h.on_new_tick(make_bar(200))

    # latest snapshot = get_snapshot(ts) where ts >= last timestamp
    snap = h.get_snapshot(999)
    assert isinstance(snap, OHLCVSnapshot)
    assert snap.timestamp == 200


def test_realtime_handler_get_snapshot():
    h = RealTimeDataHandler("BTC")
    h.on_new_tick(make_bar(100))
    h.on_new_tick(make_bar(200))

    snap = h.get_snapshot(150)
    assert isinstance(snap, OHLCVSnapshot)
    assert snap.timestamp == 100  # anti-lookahead


def test_realtime_handler_window():
    h = RealTimeDataHandler("BTC")
    for ts in [100, 200, 300]:
        h.on_new_tick(make_bar(ts))

    w = h.window(250, 5)
    assert len(w) == 2
    assert w["timestamp"].tolist() == [100, 200]


def test_realtime_reset():
    h = RealTimeDataHandler("BTC")
    h.on_new_tick(make_bar(100))
    h.reset()

    assert h.get_snapshot(999) is None
    assert h.last_timestamp() is None