import time
from contextlib import contextmanager
from .logger import get_logger, log_debug


@contextmanager
def timed_block(name: str, level="debug"):
    """Profile execution time of a code block."""
    logger = get_logger()

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000  # ms
        log_debug(logger, f"[TIMER] {name}", elapsed_ms=round(elapsed, 3))



"""
Example usage:
from quant_engine.utils.timer import timed_block
from quant_engine.utils.logger import get_logger, log_info

logger = get_logger()

with timed_block("feature_extraction"):
    features = extractor.compute(data)

log_info(logger, "Features computed", rows=len(features))
"""