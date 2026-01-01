from __future__ import annotations

# Requires local cleaned dataset; skipped in CI.

from datetime import datetime, timezone
from pathlib import Path

import asyncio
import pytest

from ingestion.contracts.tick import IngestionTick, _coerce_epoch_ms
from ingestion.ohlcv.normalize import BinanceOHLCVNormalizer
from ingestion.ohlcv.source import OHLCVFileSource
from ingestion.ohlcv.worker import OHLCVWorker
from ingestion.option_chain.normalize import DeribitOptionChainNormalizer
from ingestion.option_chain.source import OptionChainFileSource
from ingestion.option_chain.worker import OptionChainWorker
from quant_engine.data.ohlcv.realtime import OHLCVDataHandler
from quant_engine.data.derivatives.option_chain.chain_handler import OptionChainDataHandler


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

START_MS_OH = _to_ms(datetime(2023, 12, 1, 22, 0, 0, tzinfo=timezone.utc))
END_MS_OH = _to_ms(datetime(2024, 2, 28, 3, 0, 0, tzinfo=timezone.utc))

START_MS = _to_ms(datetime(2025, 12, 29, 22, 0, 0, tzinfo=timezone.utc))
END_MS = _to_ms(datetime(2025, 12, 30, 3, 0, 0, tzinfo=timezone.utc))


@pytest.mark.local
@pytest.mark.asyncio
async def test_replay_semantics_ohlcv_15m_btcusdt(monkeypatch: pytest.MonkeyPatch) -> None:
    path = Path("data/cleaned/ohlcv/BTCUSDT/15m/2025.parquet")
    if not path.exists():
        pytest.skip(f"Missing local data file: {path}")

    source = OHLCVFileSource(
        root=Path("data/cleaned/ohlcv"),
        symbol="BTCUSDT",
        interval="15m",
        start_ts=START_MS_OH,
        end_ts=END_MS_OH,
    )

    normalizer = BinanceOHLCVNormalizer(symbol="BTCUSDT")
    worker = OHLCVWorker(
        source=source,
        normalizer=normalizer,
        symbol="BTCUSDT",
        interval="15m",
        poll_interval_ms=1,
    )

    orig_sleep = asyncio.sleep
    async def fast_sleep(_seconds: float) -> None:
        await orig_sleep(0)


    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    emitted: list[IngestionTick] = []

    async def emit(tick: IngestionTick) -> None:
        emitted.append(tick)

    await worker.run(emit)

    windowed = [t for t in emitted if START_MS_OH <= int(t.data_ts) <= END_MS_OH]
    if len(windowed) < 5:
        pytest.skip("Not enough OHLCV bars in window for test")
    emitted = windowed
    assert all(isinstance(t, IngestionTick) for t in emitted)
    assert all(t.domain == "ohlcv" for t in emitted)
    assert all(t.symbol == "BTCUSDT" for t in emitted)
    data_ts = [t.data_ts for t in emitted]
    assert data_ts == sorted(data_ts)
    assert all("open" in t.payload and "high" in t.payload and "low" in t.payload and "close" in t.payload for t in emitted)

    handler = OHLCVDataHandler("BTCUSDT", interval="15m", cache={"maxlen": 10_000})
    anchors = [data_ts[0], data_ts[len(data_ts) // 2], data_ts[-1]]
    for anchor_ts in anchors:
        handler.align_to(anchor_ts)
        for tick in emitted:
            if tick.data_ts <= anchor_ts:
                handler.on_new_tick(tick)
        assert handler.last_timestamp() is None or handler.last_timestamp() <= anchor_ts # type: ignore
        snap = handler.get_snapshot(anchor_ts)
        assert snap is None or snap.data_ts <= anchor_ts
        dfw = handler.window(anchor_ts, n=10)
        if not dfw.empty and "data_ts" in dfw.columns:
            assert int(dfw["data_ts"].max()) <= anchor_ts


@pytest.mark.local
@pytest.mark.asyncio
async def test_replay_semantics_option_chain_1m_btc_parallel_stability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = Path("data/cleaned/option_chain/BTC/1m/2025")
    p1 = base / "2025_12_29.parquet"
    p2 = base / "2025_12_30.parquet"
    if not p1.exists() or not p2.exists():
        pytest.skip(f"Missing local data files: {p1} or {p2}")

    option_source = OptionChainFileSource(
        root=Path("data/cleaned/option_chain"),
        asset="BTC",
        interval="1m",
        start_ts=START_MS,
        end_ts=END_MS,
    )

    ohlcv_path = Path("data/cleaned/ohlcv/BTCUSDT/15m/2025.parquet")
    if not ohlcv_path.exists():
        pytest.skip(f"Missing local data file: {ohlcv_path}")
    ohlcv_source = OHLCVFileSource(
        root=Path("data/cleaned/ohlcv"),
        symbol="BTCUSDT",
        interval="15m",
        start_ts=START_MS_OH,
        end_ts=END_MS_OH,
    )

    option_normalizer = DeribitOptionChainNormalizer(symbol="BTC")
    option_worker = OptionChainWorker(
        source=option_source,
        normalizer=option_normalizer,
        symbol="BTC",
        poll_interval_ms=0,
    )

    ohlcv_normalizer = BinanceOHLCVNormalizer(symbol="BTCUSDT")
    ohlcv_worker = OHLCVWorker(
        source=ohlcv_source,
        normalizer=ohlcv_normalizer,
        symbol="BTCUSDT",
        interval="15m",
        poll_interval_ms=1,
    )

    orig_sleep = asyncio.sleep
    async def fast_sleep(_seconds: float) -> None:
        await orig_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    option_emitted: list[IngestionTick] = []
    ohlcv_emitted: list[IngestionTick] = []

    async def emit_option(tick: IngestionTick) -> None:
        option_emitted.append(tick)

    async def emit_ohlcv(tick: IngestionTick) -> None:
        ohlcv_emitted.append(tick)

    await asyncio.wait_for(
        asyncio.gather(
            option_worker.run(emit_option),
            ohlcv_worker.run(emit_ohlcv),
        ),
        timeout=10.0,
    )

    option_emitted = [t for t in option_emitted if START_MS <= int(t.data_ts) <= END_MS]
    ohlcv_emitted = [t for t in ohlcv_emitted if START_MS_OH <= int(t.data_ts) <= END_MS_OH]
    if len(option_emitted) < 5 or len(ohlcv_emitted) < 5:
        pytest.skip("Not enough ticks in window for parallel stability test")
    assert all(isinstance(t, IngestionTick) for t in option_emitted)
    assert all(isinstance(t, IngestionTick) for t in ohlcv_emitted)
    assert [t.data_ts for t in option_emitted] == sorted(t.data_ts for t in option_emitted)
    assert [t.data_ts for t in ohlcv_emitted] == sorted(t.data_ts for t in ohlcv_emitted)

    chain_handler = OptionChainDataHandler("BTC")
    anchors = [
        option_emitted[0].data_ts,
        option_emitted[len(option_emitted) // 2].data_ts,
        option_emitted[-1].data_ts,
    ]
    for anchor_ts in anchors:
        chain_handler.align_to(anchor_ts)
        for tick in option_emitted:
            if tick.data_ts <= anchor_ts:
                chain_handler.on_new_tick(tick)
        snap = chain_handler.get_snapshot(anchor_ts)
        assert snap is None or int(snap.data_ts) <= anchor_ts
