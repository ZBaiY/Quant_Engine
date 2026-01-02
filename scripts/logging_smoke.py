#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from quant_engine.utils.logger import (
    get_logger,
    init_logging,
    log_data_integrity,
    log_debug,
    log_exception,
    log_info,
    log_warn,
)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_temp_config(config: dict, temp_dir: Path) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / "logging_smoke.json"
    temp_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return temp_path


def _resolve_file_path(template: str, run_id: str, mode: str) -> str:
    return str(Path(template.format(run_id=run_id, mode=mode)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant Engine logging smoke script.")
    parser.add_argument("--mode", default=None, help="logging profile name")
    parser.add_argument("--run-id", default=None, help="run id for log context")
    parser.add_argument("--enable-file", action="store_true", default=True, help="enable file logging")
    parser.add_argument("--file-path", default=None, help="file path template override")
    parser.add_argument("--n", type=int, default=5, help="number of sample info logs")
    args = parser.parse_args()

    config_path = Path("configs/logging.json").resolve()
    cfg = _load_config(config_path)

    profiles = cfg.get("profiles", {})
    if not isinstance(profiles, dict):
        raise TypeError("logging.json 'profiles' must be a dict")

    active_profile = cfg.get("active_profile") or "default"
    mode = args.mode or str(active_profile)
    if mode not in profiles:
        raise KeyError(f"logging profile not found: {mode}")

    run_id = args.run_id or f"smoke_{int(time.time())}"
    profile = dict(profiles.get(mode, {}))
    file_cfg = profile.get("handlers", {}).get("file", {})
    if not isinstance(file_cfg, dict):
        file_cfg = {}
    if args.enable_file:
        template = args.file_path or "artifacts/runs/test_{run_id}/logs/testlogger_{mode}.jsonl"
        file_cfg = dict(file_cfg)
        file_cfg["enabled"] = True
        file_cfg["path"] = template
    else:
        file_cfg = dict(file_cfg)
        file_cfg["enabled"] = False

    handlers_cfg = dict(profile.get("handlers", {}))
    handlers_cfg["file"] = file_cfg
    profile["handlers"] = handlers_cfg

    profiles[mode] = profile
    cfg["profiles"] = profiles
    
    with tempfile.TemporaryDirectory(prefix="qe_logging_smoke_") as temp_dir:
        temp_config = _write_temp_config(cfg, Path(temp_dir))
        init_logging(config_path=str(temp_config), run_id=run_id, mode=mode)

        logger = get_logger("quant_engine.smoke")

        for idx in range(args.n):
            log_info(logger, "smoke.start", sample=idx + 1, total=args.n)

        log_warn(logger, "smoke.warn", detail="watchlist")

        try:
            raise ValueError("smoke exception")
        except ValueError:
            log_exception(logger, "smoke.exception", detail="handled")

        log_data_integrity(
            logger,
            "data.gap_detected",
            symbol="BTCUSDT",
            gap_type="missing",
        )
        log_debug(logger, "smoke.debug", detail="debug path")

        print(f"logging mode: {mode}")
        if file_cfg.get("enabled"):
            resolved = _resolve_file_path(file_cfg["path"], run_id, mode)
            Path(resolved).parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            print(f"file logging: enabled -> {resolved}")
            print(f"hint: tail -f {resolved}")
        else:
            print("file logging: disabled (stdout only)")


if __name__ == "__main__":
    main()
