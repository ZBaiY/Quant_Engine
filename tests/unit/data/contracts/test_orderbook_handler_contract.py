import pytest
import pandas as pd

from quant_engine.data.orderbook.cache import OrderbookCache
from quant_engine.data.orderbook.loader import OrderbookLoader
from quant_engine.data.orderbook.realtime import RealTimeOrderbookHandler
from quant_engine.data.orderbook.snapshot import OrderbookSnapshot


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def make_snapshot(ts, bb=100, bb_sz=1, ba=101, ba_sz=2, bids=None, asks=None):
    return OrderbookSnapshot(
        timestamp=float(ts),
        symbol="BTC",
        best_bid=float(bb),
        best_bid_size=float(bb_sz),
        best_ask=float(ba),
        best_ask_size=float(ba_sz),
        bids=bids or [],
        asks=asks or [],
        latency=0.0,
    )


# ------------------------------------------------------------
# OrderbookCache Contract Tests
# ------------------------------------------------------------
def test_cache_update_and_latest():
    cache = OrderbookCache(window=5)
    snap1 = make_snapshot(100)
    snap2 = make_snapshot(200)

    cache.update(snap1)
    cache.update(snap2)

    latest = cache.get_snapshot()
    assert latest is not None
    assert latest.timestamp == 200


def test_cache_window_before_ts():
    cache = OrderbookCache()
    cache.update(make_snapshot(100))
    cache.update(make_snapshot(200))
    cache.update(make_snapshot(300))

    snaps = cache.window_before_ts(250, n=5)
    assert [s.timestamp for s in snaps] == [100, 200]


def test_cache_latest_before_ts():
    cache = OrderbookCache()
    cache.update(make_snapshot(100))
    cache.update(make_snapshot(200))
    cache.update(make_snapshot(300))

    s = cache.latest_before_ts(250)
    assert s is not None
    assert s.timestamp == 200

    assert cache.latest_before_ts(50) is None


def test_cache_has_ts():
    cache = OrderbookCache()
    cache.update(make_snapshot(100))
    assert cache.has_ts(150) is True
    assert cache.has_ts(50) is False


def test_cache_clear():
    cache = OrderbookCache()
    cache.update(make_snapshot(100))
    cache.clear()
    assert cache.get_snapshot() is None


# ------------------------------------------------------------
# OrderbookSnapshot Contract Tests
# ------------------------------------------------------------
def test_snapshot_mid_and_spread():
    s = make_snapshot(100, bb=99, ba=101)
    assert s.mid_price() == pytest.approx(100)
    assert s.spread() == pytest.approx(2)


def test_snapshot_depth_parsing():
    s = make_snapshot(
        100,
        bids=[{"price": 99.0, "qty": 1.0}, {"price": 98.0, "qty": 2.0}],
        asks=[{"price": 101.0, "qty": 3.0}, {"price": 102.0, "qty": 4.0}],
    )

    # bids
    assert len(s.bids) == 2
    assert s.bids[0]["price"] == 99.0
    assert s.bids[0]["qty"] == 1.0
    assert s.bids[1]["price"] == 98.0
    assert s.bids[1]["qty"] == 2.0

    # asks
    assert len(s.asks) == 2
    assert s.asks[0]["price"] == 101.0
    assert s.asks[0]["qty"] == 3.0
    assert s.asks[1]["price"] == 102.0
    assert s.asks[1]["qty"] == 4.0


def test_snapshot_from_dataframe():
    df = pd.DataFrame([
        {
            "timestamp": 100,
            "best_bid": 99,
            "best_bid_size": 1,
            "best_ask": 101,
            "best_ask_size": 2,
            "bids": [],
            "asks": []
        }
    ])
    snap = OrderbookSnapshot.from_dataframe(df, ts=120, symbol="BTC")
    assert snap.timestamp == 100
    assert snap.latency == pytest.approx(20)


# ------------------------------------------------------------
# OrderbookLoader Contract Tests
# ------------------------------------------------------------
def test_loader_standardize_dataframe():
    df = pd.DataFrame([
        {
            "timestamp": 100,
            "best_bid": 99,
            "best_ask": 101,
            "bids": [{"price": 98, "qty": 1}],
            "asks": [{"price": 102, "qty": 1}],
        },
        {
            "timestamp": 90,
            "best_bid": 95,
            "best_ask": 105,
            "bids": [],
            "asks": [],
        }
    ])

    out = OrderbookLoader.from_dataframe(df)
    assert isinstance(out, list)
    assert out[0]["timestamp"] == 90   # sorted
    assert isinstance(out[0]["best_bid"], float)


# ------------------------------------------------------------
# RealTimeOrderbookHandler Contract Tests
# ------------------------------------------------------------


# v4-compliant: latest snapshot should be retrieved via get_snapshot(ts)
def test_realtime_latest_via_snapshot_contract():
    handler = RealTimeOrderbookHandler("BTC")
    handler.on_new_snapshot(make_snapshot(100))
    handler.on_new_snapshot(make_snapshot(200))

    # According to v4 design, latest snapshot should be retrieved via get_snapshot(ts)
    snap = handler.get_snapshot(999)   # ts >> latest timestamp
    assert snap is not None
    assert snap.timestamp == 200


def test_realtime_get_snapshot_alignment():
    handler = RealTimeOrderbookHandler("BTC")
    handler.on_new_snapshot(make_snapshot(100))
    handler.on_new_snapshot(make_snapshot(200))

    snap = handler.get_snapshot(150)
    assert snap is not None
    assert snap.timestamp == 100

    snap2 = handler.get_snapshot(1000)
    assert snap2 is not None
    assert snap2.timestamp == 200


def test_realtime_window_aligned():
    handler = RealTimeOrderbookHandler("BTC")

    for ts in [100, 200, 300]:
        handler.on_new_snapshot(make_snapshot(ts))

    w = handler.window(ts=250, n=5)
    assert [s.timestamp for s in w] == [100, 200]


def test_realtime_reset():
    handler = RealTimeOrderbookHandler("BTC")
    handler.on_new_snapshot(make_snapshot(100))
    handler.reset()
    # v4: latest snapshot must be retrieved via get_snapshot(ts)
    assert handler.get_snapshot(999) is None
    assert handler.last_timestamp() is None
