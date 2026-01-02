from __future__ import annotations

import pytest

from quant_engine.runtime.lifecycle import LifecycleGuard, RuntimePhase


def test_lifecycle_guard_happy_path() -> None:
    guard = LifecycleGuard()
    guard.enter(RuntimePhase.PRELOAD)
    guard.enter(RuntimePhase.WARMUP)
    guard.enter(RuntimePhase.INGEST)
    guard.enter(RuntimePhase.STEP)
    guard.enter(RuntimePhase.INGEST)
    guard.enter(RuntimePhase.STEP)
    guard.enter(RuntimePhase.FINISH)


def test_lifecycle_guard_rejects_invalid_start() -> None:
    guard = LifecycleGuard()
    with pytest.raises(RuntimeError):
        guard.enter(RuntimePhase.WARMUP)


def test_lifecycle_guard_rejects_invalid_transition() -> None:
    guard = LifecycleGuard()
    guard.enter(RuntimePhase.PRELOAD)
    guard.enter(RuntimePhase.WARMUP)
    with pytest.raises(RuntimeError):
        guard.enter(RuntimePhase.STEP)
