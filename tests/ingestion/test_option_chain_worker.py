from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

import ingestion.option_chain.source as option_chain_source
from ingestion.contracts.tick import IngestionTick
from ingestion.option_chain.normalize import DeribitOptionChainNormalizer
from ingestion.option_chain.source import OptionChainFileSource, DeribitOptionChainRESTSource
from ingestion.option_chain.worker import OptionChainWorker
from ingestion.contracts.tick import _to_interval_ms


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


@pytest.mark.asyncio
async def test_option_chain_worker_run_emits_ticks_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(option_chain_source, "DATA_ROOT", tmp_path)

    root = tmp_path / "raw" / "option_chain"
    asset = "BTC"
    interval = "1m"
    base = root / asset / interval
    df = pd.DataFrame(
        [
            {
                "arrival_ts": 1_700_000_000_000,
                "instrument_name": "BTC-1JAN24-10000-C",
                "expiration_timestamp": 1_700_100_000_000,
                "strike": 10_000,
                "option_type": "call",
            },
            {
                "arrival_ts": 1_700_000_001_000,
                "instrument_name": "BTC-1JAN24-10000-P",
                "expiration_timestamp": 1_700_100_000_000,
                "strike": 10_000,
                "option_type": "put",
            },
        ]
    )
    _write_parquet(df, base / "2024.parquet")

    source = OptionChainFileSource(root=root, asset=asset, interval=interval)
    normalizer = DeribitOptionChainNormalizer(symbol=asset)
    interval_ms = _to_interval_ms(interval)
    worker = OptionChainWorker(
        normalizer=normalizer,
        source=source,
        symbol=asset,
        interval=interval,
        interval_ms=int(interval_ms) if interval_ms is not None else None,
        poll_interval_ms=1,
    )

    async def fast_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    emitted: list[IngestionTick] = []

    async def emit(tick: IngestionTick) -> None:
        emitted.append(tick)

    await worker.run(emit)

    assert [tick.data_ts for tick in emitted] == [1_700_000_000_000, 1_700_000_001_000]
    assert all(isinstance(tick, IngestionTick) for tick in emitted)


def test_option_chain_worker_backfill_persists_frame_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(option_chain_source, "DATA_ROOT", tmp_path)

    asset = "BTC"
    interval = "1m"
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
    data_ts = 1_700_000_000_000

    class _FetchSource:
        interval = "1m"

        def backfill(self, start_ts: int, end_ts: int):
            return [{"data_ts": data_ts, "frame": df}]

    source = OptionChainFileSource(root=tmp_path / "raw" / "option_chain", asset=asset, interval=interval)
    normalizer = DeribitOptionChainNormalizer(symbol=asset)
    worker = OptionChainWorker(
        normalizer=normalizer,
        source=source,
        fetch_source=cast(DeribitOptionChainRESTSource, _FetchSource()),
        symbol=asset,
        interval=interval,
    )

    count = worker.backfill(start_ts=data_ts, end_ts=data_ts, anchor_ts=data_ts)

    assert count == 1


def _count_parquet_files(root: Path) -> int:
    return len(list(root.rglob("*.parquet"))) if root.exists() else 0


def test_option_chain_worker_backfill_skips_empty_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(option_chain_source, "DATA_ROOT", tmp_path)

    asset = "BTC"
    interval = "1m"
    data_ts = 1_700_000_000_000

    class _FetchSource:
        interval = "1m"

        def backfill(self, start_ts: int, end_ts: int):
            return [{"data_ts": data_ts, "records": []}]

    source = OptionChainFileSource(root=tmp_path / "raw" / "option_chain", asset=asset, interval=interval)
    normalizer = DeribitOptionChainNormalizer(symbol=asset)
    worker = OptionChainWorker(
        normalizer=normalizer,
        source=source,
        fetch_source=cast(DeribitOptionChainRESTSource, _FetchSource()),
        symbol=asset,
        interval=interval,
    )

    count = worker.backfill(start_ts=data_ts, end_ts=data_ts, anchor_ts=data_ts)

    assert count == 0
    assert _count_parquet_files(tmp_path / "raw" / "option_chain") == 0


def test_option_chain_worker_backfill_skips_empty_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(option_chain_source, "DATA_ROOT", tmp_path)

    asset = "BTC"
    interval = "1m"
    data_ts = 1_700_000_000_000
    empty_df = pd.DataFrame([])

    class _FetchSource:
        interval = "1m"

        def backfill(self, start_ts: int, end_ts: int):
            return [{"data_ts": data_ts, "frame": empty_df}]

    source = OptionChainFileSource(root=tmp_path / "raw" / "option_chain", asset=asset, interval=interval)
    normalizer = DeribitOptionChainNormalizer(symbol=asset)
    worker = OptionChainWorker(
        normalizer=normalizer,
        source=source,
        fetch_source=cast(DeribitOptionChainRESTSource, _FetchSource()),
        symbol=asset,
        interval=interval,
    )

    count = worker.backfill(start_ts=data_ts, end_ts=data_ts, anchor_ts=data_ts)

    assert count == 0
    assert _count_parquet_files(tmp_path / "raw" / "option_chain") == 0
