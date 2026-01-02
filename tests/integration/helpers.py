from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def find_ohlcv_root(data_root: Path, symbol: str, interval: str) -> Path | None:
    candidates = [
        data_root / "cleaned" / "ohlcv",
        data_root / "raw" / "ohlcv",
    ]
    for root in candidates:
        base = root / symbol / interval
        if not base.exists():
            continue
        if list(base.rglob("*.parquet")):
            return root
    return None


def earliest_ohlcv_ts_ms(path: Path) -> int | None:
    if not path.exists():
        return None
    files = list(path.rglob("*.parquet"))
    if not files:
        return None
    candidates: list[datetime] = []
    for fp in files:
        stem = fp.stem
        if re.fullmatch(r"\\d{4}", stem):
            candidates.append(datetime(int(stem), 1, 1, tzinfo=timezone.utc))
            continue
        m = re.search(r"(\\d{4})_(\\d{2})_(\\d{2})", stem)
        if m:
            y, mo, d = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            candidates.append(datetime(y, mo, d, tzinfo=timezone.utc))
    if not candidates:
        return None
    return int(min(candidates).timestamp() * 1000)
