from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import ingestion.option_chain.source as option_chain_source
from ingestion.contracts.tick import IngestionTick
from ingestion.option_chain.source import OptionChainFileSource


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_option_chain_file_source_single_file_coerces_seconds(
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
                "arrival_ts": 1_700_000_000,
                "instrument_name": "BTC-1JAN24-10000-C",
                "expiry_ts": 1_700_100_000_000,
                "strike": 10_000,
                "option_type": "call",
                "delta": 0.5,
            },
            {
                "arrival_ts": 1_700_000_001.5,
                "instrument_name": "BTC-1JAN24-10000-P",
                "expiry_ts": 1_700_100_000_000,
                "strike": 10_000,
                "option_type": "put",
                "delta": -0.5,
            },
        ]
    )
    _write_parquet(df, base / "2024.parquet")

    src = OptionChainFileSource(root=root, asset=asset, interval=interval)
    rows = list(src)

    assert [r["arrival_ts"] for r in rows] == [1_700_000_000_000, 1_700_000_001_500]
    assert [r["data_ts"] for r in rows] == [1_700_000_000_000, 1_700_000_001_500]
    assert all(isinstance(r["arrival_ts"], int) for r in rows)
    assert isinstance(rows[0]["frame"], pd.DataFrame)
    assert "delta" in rows[0]["frame"].columns
    assert all(not isinstance(r, IngestionTick) for r in rows)


def test_option_chain_file_source_multiple_files_monotonic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(option_chain_source, "DATA_ROOT", tmp_path)
    root = tmp_path / "raw" / "option_chain"
    asset = "BTC"
    interval = "1m"
    base = root / asset / interval
    df_2023 = pd.DataFrame(
        [
            {
                "arrival_ts": 1_700_000_000_000,
                "instrument_name": "BTC-1JAN24-10000-C",
                "expiry_ts": 1_700_100_000_000,
                "strike": 10_000,
                "option_type": "call",
            }
        ]
    )
    df_2024 = pd.DataFrame(
        [
            {
                "arrival_ts": 1_700_000_001_000,
                "instrument_name": "BTC-1JAN24-10000-P",
                "expiry_ts": 1_700_100_000_000,
                "strike": 10_000,
                "option_type": "put",
            }
        ]
    )
    _write_parquet(df_2023, base / "2023.parquet")
    _write_parquet(df_2024, base / "2024.parquet")

    src = OptionChainFileSource(root=root, asset=asset, interval=interval)
    rows = list(src)

    assert [r["arrival_ts"] for r in rows] == [1_700_000_000_000, 1_700_000_001_000]
    assert [r["arrival_ts"] for r in rows] == sorted(r["arrival_ts"] for r in rows)


def test_option_chain_file_source_missing_data_ts_raises(
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
                "instrument_name": "BTC-1JAN24-10000-C",
                "expiry_ts": 1_700_100_000_000,
                "strike": 10_000,
                "option_type": "call",
            }
        ]
    )
    _write_parquet(df, base / "2024.parquet")

    src = OptionChainFileSource(root=root, asset=asset, interval=interval)
    with pytest.raises(ValueError, match="missing arrival_ts"):
        list(src)


def test_option_chain_file_source_invalid_data_ts_raises(
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
                "arrival_ts": None,
                "instrument_name": "BTC-1JAN24-10000-C",
                "expiry_ts": 1_700_100_000_000,
                "strike": 10_000,
                "option_type": "call",
            }
        ]
    )
    _write_parquet(df, base / "2024.parquet")

    src = OptionChainFileSource(root=root, asset=asset, interval=interval)
    with pytest.raises(ValueError, match="timestamp"):
        list(src)
