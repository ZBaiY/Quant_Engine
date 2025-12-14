import logging
import json
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------
# Base class: artifact-grade log sink
# ---------------------------------------------------------------------

class ArtifactFileHandler(logging.Handler):
    """
    Base class for artifact logs.
    Subclasses decide which category they accept.
    """

    category: str  # must be overridden

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        context = getattr(record, "context", None)
        if not isinstance(context, dict):
            return

        if context.get("category") != self.category:
            return

        event: dict[str, Any] = {
            "ts": record.created,          # numeric ts for replay
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
            "context": context,
        }

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------

class DecisionFileHandler(ArtifactFileHandler):
    """
    Stores decision_trace events.
    """
    category = "decision_trace"

    def __init__(self, run_id: str, base_dir: str = "artifacts/runs"):
        path = (
            Path(base_dir)
            / run_id
            / "decisions"
            / "decisions.jsonl"
        )
        super().__init__(path)


class ExecutionFileHandler(ArtifactFileHandler):
    """
    Stores execution_discrepancy events.
    """
    category = "execution_discrepancy"

    def __init__(self, run_id: str, base_dir: str = "artifacts/runs"):
        path = (
            Path(base_dir)
            / run_id
            / "execution"
            / "execution.jsonl"
        )
        super().__init__(path)


class DataRepairFileHandler(ArtifactFileHandler):
    """
    Stores data integrity repair / backfill events.
    """
    category = "data_integrity"

    def __init__(self, run_id: str, base_dir: str = "artifacts/runs"):
        path = (
            Path(base_dir)
            / run_id
            / "data_repairs"
            / "repairs.jsonl"
        )
        super().__init__(path)


# ---------------------------------------------------------------------
# Wiring helper (optional but clean)
# ---------------------------------------------------------------------

def attach_artifact_handlers(
    logger: logging.Logger,
    *,
    run_id: str,
    decisions: bool = True,
    execution: bool = True,
    data_repairs: bool = False,
    base_dir: str = "artifacts/runs",
) -> None:
    """
    Attach artifact handlers to a logger at runtime.
    Call once during engine bootstrap.
    """

    if decisions:
        logger.addHandler(
            DecisionFileHandler(run_id, base_dir=base_dir)
        )

    if execution:
        logger.addHandler(
            ExecutionFileHandler(run_id, base_dir=base_dir)
        )

    if data_repairs:
        logger.addHandler(
            DataRepairFileHandler(run_id, base_dir=base_dir)
        )