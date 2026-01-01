from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

import ingestion.option_chain.source as option_chain_source
from ingestion.contracts.tick import IngestionTick
from ingestion.option_chain.normalize import DeribitOptionChainNormalizer
from ingestion.option_chain.source import OptionChainFileSource
from ingestion.option_chain.worker import OptionChainWorker


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
                "data_ts": 1_700_000_000_000,
                "instrument_name": "BTC-1JAN24-10000-C",
                "expiration_timestamp": 1_700_100_000_000,
                "strike": 10_000,
                "option_type": "call",
            },
            {
                "data_ts": 1_700_000_001_000,
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
    worker = OptionChainWorker(normalizer=normalizer, source=source, symbol=asset, poll_interval_ms=1)

    async def fast_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    emitted: list[IngestionTick] = []

    async def emit(tick: IngestionTick) -> None:
        emitted.append(tick)

    await worker.run(emit)

    assert [tick.data_ts for tick in emitted] == [1_700_000_000_000, 1_700_000_001_000]
    assert all(isinstance(tick, IngestionTick) for tick in emitted)
