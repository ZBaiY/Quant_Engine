from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import ingestion.ohlcv.source as ohlcv_source
from ingestion.contracts.tick import IngestionTick
from ingestion.ohlcv.source import OHLCVFileSource


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_ohlcv_file_source_single_file_with_data_ts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ohlcv_source, "DATA_ROOT", tmp_path)
    root = tmp_path / "ohlcv"
    symbol = "BTCUSDT"
    interval = "1m"
    base = root / symbol / interval
    df = pd.DataFrame(
        [
            {"data_ts": 1_700_000_000, "open": 2.0, "close": 2.2},
            {"data_ts": 1_700_000_001_000, "open": 1.0, "close": 1.1},
        ]
    )
    _write_parquet(df, base / "2024.parquet")

    src = OHLCVFileSource(root=root, symbol=symbol, interval=interval)
    rows = list(src)

    assert [r["data_ts"] for r in rows] == [1_700_000_000_000, 1_700_000_001_000]
    assert all(isinstance(r["data_ts"], int) for r in rows)
    assert all(not isinstance(r, IngestionTick) for r in rows)


def test_ohlcv_file_source_infers_from_close_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ohlcv_source, "DATA_ROOT", tmp_path)
    root = tmp_path / "ohlcv"
    symbol = "BTCUSDT"
    interval = "1m"
    base = root / symbol / interval
    df = pd.DataFrame(
        [
            {"close_time": 1_700_000_003, "open": 1.0},
            {"close_time": 1_700_000_001.5, "open": 2.0},
        ]
    )
    _write_parquet(df, base / "2024.parquet")

    src = OHLCVFileSource(root=root, symbol=symbol, interval=interval)
    rows = list(src)

    assert [r["data_ts"] for r in rows] == [1_700_000_001_500, 1_700_000_003_000]
    assert all(isinstance(r["data_ts"], int) for r in rows)


def test_ohlcv_file_source_multiple_files_infer_from_open_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ohlcv_source, "DATA_ROOT", tmp_path)
    root = tmp_path / "ohlcv"
    symbol = "BTCUSDT"
    interval = "1m"
    base = root / symbol / interval
    base_ts = 1_700_000_000
    df_2023 = pd.DataFrame(
        [
            {"open_time": base_ts, "open": 1.0},
            {"open_time": base_ts + 60, "open": 1.1},
        ]
    )
    df_2024 = pd.DataFrame(
        [
            {"open_time": base_ts + 120, "open": 1.2},
            {"open_time": base_ts + 180, "open": 1.3},
        ]
    )
    _write_parquet(df_2023, base / "2023.parquet")
    _write_parquet(df_2024, base / "2024.parquet")

    src = OHLCVFileSource(root=root, symbol=symbol, interval=interval)
    rows = list(src)

    data_ts = [r["data_ts"] for r in rows]
    assert data_ts == [
        (base_ts * 1000) + 60_000,
        (base_ts * 1000) + 120_000,
        (base_ts * 1000) + 180_000,
        (base_ts * 1000) + 240_000,
    ]
    assert all(isinstance(ts, int) for ts in data_ts)


def test_ohlcv_file_source_missing_timestamp_columns_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ohlcv_source, "DATA_ROOT", tmp_path)
    root = tmp_path / "ohlcv"
    symbol = "BTCUSDT"
    interval = "1m"
    base = root / symbol / interval
    df = pd.DataFrame([{"open": 1.0, "close": 1.1}])
    _write_parquet(df, base / "2024.parquet")

    src = OHLCVFileSource(root=root, symbol=symbol, interval=interval)
    with pytest.raises(ValueError, match="missing 'data_ts'"):
        list(src)


def test_ohlcv_file_source_invalid_timestamps_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ohlcv_source, "DATA_ROOT", tmp_path)
    root = tmp_path / "ohlcv"
    symbol = "BTCUSDT"
    interval = "1m"
    base = root / symbol / interval
    df = pd.DataFrame([{"data_ts": None, "open": 1.0, "close": 1.1}])
    _write_parquet(df, base / "2024.parquet")

    src = OHLCVFileSource(root=root, symbol=symbol, interval=interval)
    with pytest.raises(ValueError, match="timestamp"):
        list(src)
