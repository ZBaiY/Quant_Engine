from __future__ import annotations

from pathlib import Path
from typing import Tuple


def repo_root_from_file(file: str | Path, *, levels_up: int) -> Path:
    if levels_up < 0:
        raise ValueError("levels_up must be >= 0")
    return Path(file).resolve().parents[levels_up]


def data_root_from_file(file: str | Path, *, levels_up: int) -> Path:
    return repo_root_from_file(file, levels_up=levels_up) / "data"


def artifacts_root_from_file(file: str | Path, *, levels_up: int) -> Path:
    return repo_root_from_file(file, levels_up=levels_up) / "artifacts"


def resolve_under_root(root: Path, p: str | Path, *, strip_prefix: str | None = None) -> Path:
    root = Path(root)
    candidate = Path(p)
    if candidate.is_absolute():
        if candidate == root or root in candidate.parents:
            return candidate
        raise ValueError(f"path must be under {root}")
    if strip_prefix:
        parts = candidate.parts
        if parts and parts[0] == strip_prefix:
            candidate = Path(*parts[1:])
    return root / candidate


def resolve_data_path(file: str | Path, p: str | Path, *, levels_up: int) -> Tuple[Path, Path]:
    root = data_root_from_file(file, levels_up=levels_up)
    return root, resolve_under_root(root, p, strip_prefix="data")


def resolve_artifacts_path(file: str | Path, p: str | Path, *, levels_up: int) -> Tuple[Path, Path]:
    root = artifacts_root_from_file(file, levels_up=levels_up)
    return root, resolve_under_root(root, p, strip_prefix="artifacts")
