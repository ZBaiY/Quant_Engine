from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from quant_engine.utils.paths import data_root_from_file, artifacts_root_from_file, repo_root_from_file

REPO_ROOT = repo_root_from_file(__file__, levels_up=1)


def _import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(name: str, root_type: str, module_file: str, levels_up: int) -> None:
    if root_type == "data":
        root = data_root_from_file(module_file, levels_up=levels_up)
    else:
        root = artifacts_root_from_file(module_file, levels_up=levels_up)
    print(f"{name}: {root_type}_root={root} exists={root.exists()}")


def main() -> None:
    modules = [
        ("apps.run_backtest", "data", 1, REPO_ROOT / "apps" / "run_backtest.py"),
        ("apps.scrap.option_chain", "data", 2, REPO_ROOT / "apps" / "scrap" / "option_chain.py"),
        ("apps.scrap.ohlcv", "data", 2, REPO_ROOT / "apps" / "scrap" / "ohlcv.py"),
        ("apps.scrap.trades", "data", 2, REPO_ROOT / "apps" / "scrap" / "trades.py"),
        ("ingestion.ohlcv.source", "data", 3, None),
        ("ingestion.orderbook.source", "data", 3, None),
        ("ingestion.trades.source", "data", 3, None),
        ("ingestion.option_chain.source", "data", 3, None),
        ("ingestion.option_trades.source", "data", 3, None),
        ("ingestion.sentiment.source", "data", 3, None),
        ("quant_engine.runtime.log_router", "artifacts", 3, None),
    ]

    for name, root_type, levels_up, path in modules:
        module_file = None
        import_err = None
        try:
            if path is None:
                module = importlib.import_module(name)
            else:
                module = _import_from_path(name, path)
            module_file = getattr(module, "__file__", None)
            if module_file is None:
                raise ImportError(f"{name} has no __file__")
        except Exception as exc:
            import_err = exc
            if path is not None:
                module_file = str(path)

        if module_file is None:
            print(f"{name}: import_failed={import_err}")
            continue

        _report(name, root_type, module_file, levels_up)
        if import_err is not None:
            print(f"{name}: import_failed={import_err}")


if __name__ == "__main__":
    main()
