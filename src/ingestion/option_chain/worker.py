from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Callable, Awaitable, Any

from ingestion.contracts.tick import IngestionTick
from ingestion.contracts.worker import IngestWorker
from ingestion.option_chain.normalize import DeribitOptionChainNormalizer
from ingestion.option_chain.source import (
    DeribitOptionChainRESTSource,
    OptionChainFileSource,
    OptionChainStreamSource,
)
from quant_engine.utils.logger import get_logger, log_info, log_warn, log_debug

_LOG_SAMPLE_EVERY = 100
_DOMAIN = "option_chain"

def _as_primitive(x: Any) -> str | int | float | bool | None:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    try:
        return str(x)
    except Exception:
        return "<unrepr>"

class OptionChainWorker(IngestWorker):
    """
    Option chain ingestion worker.
    Responsibilities:
        raw -> normalize -> emit tick
    """

    def __init__(
        self,
        *,
        normalizer: DeribitOptionChainNormalizer,
        source: OptionChainFileSource | DeribitOptionChainRESTSource | OptionChainStreamSource,
        symbol: str,
        poll_interval: float | None = None,
        poll_interval_ms: int | None = None,
        logger: logging.Logger | None = None,
    ):
        self._normalizer = normalizer
        self._source = source
        self._symbol = symbol
        self._logger = logger or get_logger(f"ingestion.{_DOMAIN}.{self.__class__.__name__}")
        self._poll_seq = 0
        self._error_logged = False
        if poll_interval_ms is not None:
            self._poll_interval_ms = int(poll_interval_ms)
        elif poll_interval is not None:
            self._poll_interval_ms = int(round(float(poll_interval) * 1000.0))
        else:
            self._poll_interval_ms = None
        if self._poll_interval_ms is not None and self._poll_interval_ms < 0:
            raise ValueError("poll_interval_ms must be >= 0")

    async def run(self, emit: Callable[[IngestionTick], Awaitable[None] | None]) -> None:
        log_info(
            self._logger,
            "ingestion.worker_start",
            worker=self.__class__.__name__,
            source_type=type(self._source).__name__,
            symbol=self._symbol,
            poll_interval_ms=self._poll_interval_ms,
            domain=_DOMAIN,
        )
        self._error_logged = False
        stop_reason = "exit"

        async def _emit(tick: IngestionTick) -> None:
            try:
                r = emit(tick)
                if inspect.isawaitable(r):
                    await r  # type: ignore[misc]
            except Exception as exc:
                self._error_logged = True
                log_warn(
                    self._logger,
                    "ingestion.emit_error",
                    worker=self.__class__.__name__,
                    symbol=self._symbol,
                    domain=_DOMAIN,
                    poll_seq=self._poll_seq,
                    err_type=type(exc).__name__,
                    err=str(exc),
                )
                raise

        try:
            # --- async source (streaming option updates) ---
            if hasattr(self._source, "__aiter__"):
                last_fetch = time.monotonic()
                async for raw in self._source:  # type: ignore
                    now = time.monotonic()
                    self._poll_seq += 1
                    if self._poll_seq % _LOG_SAMPLE_EVERY == 0:
                        log_debug(
                            self._logger,
                            "ingestion.source_fetch_success",
                            worker=self.__class__.__name__,
                            symbol=self._symbol,
                            domain=_DOMAIN,
                            latency_ms=int((now - last_fetch) * 1000),
                            n_items=1,
                            poll_seq=self._poll_seq,
                        )
                    last_fetch = time.monotonic()
                    tick = self._normalize(raw)
                    await _emit(tick)
                    # Cooperative yield: prevent starvation when the stream is bursty
                    await asyncio.sleep(0)

            # --- sync source (REST polling / file replay) ---
            else:
                last_fetch = time.monotonic()
                for raw in self._source:  # type: ignore
                    now = time.monotonic()
                    self._poll_seq += 1
                    if self._poll_seq % _LOG_SAMPLE_EVERY == 0:
                        log_debug(
                            self._logger,
                            "ingestion.source_fetch_success",
                            worker=self.__class__.__name__,
                            symbol=self._symbol,
                            domain=_DOMAIN,
                            latency_ms=int((now - last_fetch) * 1000),
                            n_items=1,
                            poll_seq=self._poll_seq,
                        )
                    last_fetch = time.monotonic()
                    tick = self._normalize(raw)
                    await _emit(tick)
                    if self._poll_interval_ms is not None:
                        await asyncio.sleep(self._poll_interval_ms / 1000.0)
                    else:
                        # Cooperative yield: prevent starvation for fast iterators
                        await asyncio.sleep(0)
        except asyncio.CancelledError:
            stop_reason = "cancelled"
            raise
        except Exception as exc:
            if not self._error_logged:
                log_warn(
                    self._logger,
                    "ingestion.source_fetch_error",
                    worker=self.__class__.__name__,
                    symbol=self._symbol,
                    domain=_DOMAIN,
                    poll_seq=self._poll_seq,
                    err_type=type(exc).__name__,
                    err=str(exc),
                    retry_count=0,
                    backoff_ms=0,
                )
            stop_reason = "error"
            raise
        finally:
            log_info(
                self._logger,
                "ingestion.worker_stop",
                worker=self.__class__.__name__,
                symbol=self._symbol,
                domain=_DOMAIN,
                reason=stop_reason,
            )

    def _normalize(self, raw: dict) -> IngestionTick:
        try:
            return self._normalizer.normalize(raw=raw)
        except Exception as exc:
            self._error_logged = True
            raw_ts = _extract_raw_ts(raw)
            log_warn(
                self._logger,
                "ingestion.normalize_drop",
                worker=self.__class__.__name__,
                symbol=self._symbol,
                domain=_DOMAIN,
                poll_seq=self._poll_seq,
                err_type=type(exc).__name__,
                err=str(exc),
                raw_type=type(raw).__name__,
                raw_ts=_as_primitive(raw_ts),
            )
            raise


def _extract_raw_ts(raw: Any) -> str | int | float | None:
    if isinstance(raw, dict):
        for key in ("data_ts", "timestamp", "ts", "time", "T", "E"):
            if key in raw:
                return raw.get(key)
    return None
