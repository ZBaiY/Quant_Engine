from __future__ import annotations

from dataclasses import asdict
from typing import Any

def assert_keys(obj: Any, keys: set[str]) -> None:
    if hasattr(obj, "__dict__"):
        d = obj.__dict__
    else:
        try:
            d = asdict(obj)
        except Exception:
            raise AssertionError(f"Object {type(obj)} is not dict-like for key assertion.")
    missing = keys - set(d.keys())
    if missing:
        raise AssertionError(f"Missing keys: {missing}. Got: {sorted(d.keys())}")
