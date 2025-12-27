#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"

mkdir -p "$ROOT/tests"/{unit,integration,smoke,helpers}
mkdir -p "$ROOT/tests/unit"/{ingestion,quant_engine}
mkdir -p "$ROOT/tests/unit/ingestion"/{contracts,ohlcv,orderbook,option_chain,option_trades,sentiment,trades}
mkdir -p "$ROOT/tests/unit/quant_engine"/{contracts,data,execution,runtime,strategy,features,risk,portfolio}
mkdir -p "$ROOT/tests/unit/quant_engine/data"/{ohlcv,orderbook,derivatives,sentiment,trades}
mkdir -p "$ROOT/tests/unit/quant_engine/data/derivatives"/{iv,option_chain,option_trades}
mkdir -p "$ROOT/tests/integration"/{pipeline,ingestion,execution}
mkdir -p "$ROOT/tests/smoke"/{backtest,mock,realtime}
mkdir -p "$ROOT/tests/helpers"

# ----------------------------
# conftest.py (common fixtures)
# ----------------------------
cat > "$ROOT/tests/conftest.py" <<'EOF'
import os
import random

import pytest

@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(0)
    os.environ.setdefault("PYTHONHASHSEED", "0")
EOF

# ----------------------------
# helpers: factories
# ----------------------------
cat > "$ROOT/tests/helpers/factories.py" <<'EOF'
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
EOF

# ----------------------------
# unit: ingestion contracts
# ----------------------------
cat > "$ROOT/tests/unit/ingestion/contracts/test_tick.py" <<'EOF'
from __future__ import annotations

import pytest

from ingestion.contracts.tick import _to_interval_ms, IngestionTick

def test_to_interval_ms_basic():
    assert _to_interval_ms("1m") == 60_000
    assert _to_interval_ms("15m") == 15 * 60_000
    assert _to_interval_ms("1h") == 60 * 60_000

@pytest.mark.parametrize("bad", ["", "0m", "1x", "m1", "1", None])
def test_to_interval_ms_rejects_bad(bad):
    with pytest.raises(Exception):
        _to_interval_ms(bad)  # type: ignore[arg-type]

def test_ingestion_tick_has_required_fields():
    t = IngestionTick(timestamp=1700000000000, data_ts=1700000000123, symbol="BTCUSDT", source="test", payload={})
    assert t.timestamp == 1700000000000
    assert t.data_ts == 1700000000123
    assert t.symbol == "BTCUSDT"
EOF

# ----------------------------
# unit: quant_engine contracts sanity
# ----------------------------
cat > "$ROOT/tests/unit/quant_engine/contracts/test_contracts_import.py" <<'EOF'
def test_contracts_importable():
    import quant_engine.contracts.feature as _  # noqa: F401
    import quant_engine.contracts.model as _    # noqa: F401
    import quant_engine.contracts.decision as _ # noqa: F401
    import quant_engine.contracts.risk as _     # noqa: F401
    import quant_engine.contracts.portfolio as _# noqa: F401
    import quant_engine.contracts.execution.order as _ # noqa: F401
EOF

# ----------------------------
# unit: execution chain minimal invariants (placeholder)
# ----------------------------
cat > "$ROOT/tests/unit/quant_engine/execution/test_execution_chain_placeholder.py" <<'EOF'
from __future__ import annotations

def test_execution_modules_importable():
    import quant_engine.execution.engine as _  # noqa: F401
    import quant_engine.execution.matching.simulated as _  # noqa: F401
    import quant_engine.execution.router.simple as _  # noqa: F401
    import quant_engine.execution.slippage.linear as _  # noqa: F401
EOF

# ----------------------------
# integration: loader->bootstrap->backtest smoke pipeline (placeholder)
# ----------------------------
cat > "$ROOT/tests/integration/pipeline/test_loader_pipeline_placeholder.py" <<'EOF'
from __future__ import annotations

def test_strategy_loader_pipeline_importable():
    import quant_engine.strategy.loader as _  # noqa: F401
    import quant_engine.runtime.bootstrap as _  # noqa: F401
EOF

# ----------------------------
# smoke: minimal backtest entry import (placeholder)
# ----------------------------
cat > "$ROOT/tests/smoke/backtest/test_run_backtest_import.py" <<'EOF'
def test_app_entry_importable():
    import apps.run_backtest as _  # noqa: F401
EOF

# ----------------------------
# README for tests (short)
# ----------------------------
cat > "$ROOT/tests/README.md" <<'EOF'
Tests layout:
- unit/: pure logic, no IO
- integration/: multi-module pipelines
- smoke/: minimal entrypoint checks
Add real tests by replacing *_placeholder.py with assertions over snapshots / determinism.
EOF

echo "[ok] tests scaffold created under: $ROOT/tests"
echo "Next: wire these into scripts/test_unit.sh and scripts/test_all.sh if needed."