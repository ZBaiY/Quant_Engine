from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from quant_engine.utils.logger import get_logger, init_logging, log_info, log_warn
from quant_engine.utils.guards import ensure_epoch_ms

def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _set_current_run(run_id: str) -> None:
    runs_dir = Path("artifacts") / "runs"
    target = runs_dir / run_id
    current = runs_dir / "_current"
    runs_dir.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    try:
        if current.exists() or current.is_symlink():
            current.unlink()
        current.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        (runs_dir / "CURRENT").write_text(run_id, encoding="utf-8")


_RUN_ID = _make_run_id()
init_logging(run_id=_RUN_ID)
_set_current_run(_RUN_ID)

from ingestion.ohlcv.worker import OHLCVWorker
from ingestion.ohlcv.source import OHLCVFileSource
from ingestion.ohlcv.normalize import BinanceOHLCVNormalizer
from ingestion.orderbook.worker import OrderbookWorker
from ingestion.orderbook.source import OrderbookFileSource
from ingestion.orderbook.normalize import BinanceOrderbookNormalizer
from ingestion.option_chain.worker import OptionChainWorker
from ingestion.option_chain.source import OptionChainFileSource
from ingestion.option_chain.normalize import DeribitOptionChainNormalizer
from ingestion.sentiment.worker import SentimentWorker
from ingestion.sentiment.source import SentimentFileSource
from ingestion.sentiment.normalize import SentimentNormalizer

from quant_engine.runtime.backtest import BacktestDriver
from quant_engine.runtime.modes import EngineMode
from quant_engine.strategy.registry import get_strategy
from quant_engine.utils.paths import data_root_from_file


logger = get_logger(__name__)
START_TS = 1622505600000  # 2021-06-01 00:00:00 UTC (epoch ms)
END_TS = 1622592000000    # 2021-06-02 00:00:00 UTC (epoch ms)
DATA_ROOT = data_root_from_file(__file__, levels_up=1)


def _has_parquet_files(path: Path) -> bool:
    if not path.exists():
        return False
    if any(path.glob("*.parquet")):
        return True
    return any(path.rglob("*.parquet"))


def _has_jsonl_files(path: Path) -> bool:
    if not path.exists():
        return False
    if any(path.glob("*.jsonl")):
        return True
    return any(path.rglob("*.jsonl"))


def _has_ohlcv_data(root: Path, *, symbol: str, interval: str) -> bool:
    return _has_parquet_files(root / symbol / interval)


def _has_orderbook_data(root: Path, *, symbol: str) -> bool:
    path = root / symbol
    return path.exists() and any(path.glob("snapshot_*.parquet"))


def _has_option_chain_data(root: Path, *, asset: str, interval: str) -> bool:
    return _has_parquet_files(root / asset / interval)


def _has_sentiment_data(root: Path, *, provider: str) -> bool:
    return _has_jsonl_files(root / provider)


async def main() -> None:
    # -------------------------------------------------
    # 1. Load & bind strategy
    # -------------------------------------------------
    StrategyCls = get_strategy("EXAMPLE")

    strategy = StrategyCls().bind(A="BTCUSDT", B="ETHUSDT")

    engine = strategy.build(mode=EngineMode.BACKTEST)
    log_info(
        logger,
        "app.engine.built",
        mode=engine.spec.mode.value,
        interval=engine.spec.interval,
        symbols=list(engine.universe.values()) if engine.universe else [],
        domains=[
            d
            for d, h in (
                ("ohlcv", engine.ohlcv_handlers),
                ("orderbook", engine.orderbook_handlers),
                ("option_chain", engine.option_chain_handlers),
                ("iv_surface", engine.iv_surface_handlers),
                ("sentiment", engine.sentiment_handlers),
                ("trades", engine.trades_handlers),
                ("option_trades", engine.option_trades_handlers),
            )
            if h
        ],
    )

    # -------------------------------------------------
    # 2. Build per-handler ingestion (generalized)
    # -------------------------------------------------
    # In backtest we stream ticks into the runtime (口径2):
    # ingestion runs fast (no throttling) and pushes ticks into a priority queue.
    tick_queue: asyncio.PriorityQueue[tuple[int, int, object]] = asyncio.PriorityQueue()
    _seq = 0

    ingestion_tasks: list[asyncio.Task[None]] = []

    async def emit_to_queue(tick: object) -> None:
        # Expect tick to have `.timestamp` (engine-time) attribute.
        nonlocal _seq
        ts = ensure_epoch_ms(getattr(tick, "timestamp"))
        await tick_queue.put((ts, _seq, tick))
        _seq += 1

    # -------------------------
    # OHLCV ingestion
    # -------------------------
    for symbol, handler in engine.ohlcv_handlers.items():
        has_local_data = _has_ohlcv_data(
            DATA_ROOT / "raw" / "ohlcv",
            symbol=symbol,
            interval=handler.interval,
        )
        if not has_local_data:
            log_warn(
                logger,
                "ingestion.worker.skipped_no_data",
                domain="ohlcv",
                symbol=symbol,
                root=str(DATA_ROOT / "raw" / "ohlcv"),
            )
            continue
        source = OHLCVFileSource(
            root=DATA_ROOT / "raw" / "ohlcv",
            symbol=symbol,
            interval=handler.interval,
            start_ts=START_TS,
            end_ts=END_TS,
        )
        normalizer = BinanceOHLCVNormalizer(symbol=symbol)
        worker = OHLCVWorker(
            source=source,
            normalizer=normalizer,
            symbol=symbol,
            poll_interval=None,  # backtest: do not throttle
        )
        log_info(
            logger,
            "ingestion.worker.start",
            domain="ohlcv",
            symbol=symbol,
            source_type=type(source).__name__,
            has_local_data=has_local_data,
            start_ts=START_TS,
            end_ts=END_TS,
        )
        ingestion_tasks.append(asyncio.create_task(worker.run(emit=emit_to_queue)))

    # -------------------------
    # Orderbook ingestion
    # -------------------------
    for symbol, handler in engine.orderbook_handlers.items():
        has_local_data = _has_orderbook_data(
            DATA_ROOT / "raw" / "orderbook",
            symbol=symbol,
        )
        if not has_local_data:
            log_warn(
                logger,
                "ingestion.worker.skipped_no_data",
                domain="orderbook",
                symbol=symbol,
                root=str(DATA_ROOT / "raw" / "orderbook"),
            )
            continue
        source = OrderbookFileSource(
            root=DATA_ROOT / "raw" / "orderbook",
            symbol=symbol,
            start_ts=START_TS,
            end_ts=END_TS,
        )
        normalizer = BinanceOrderbookNormalizer(symbol=symbol)
        worker = OrderbookWorker(
            source=source,
            normalizer=normalizer,
            symbol=symbol,
            poll_interval=None,  # backtest: do not throttle
        )
        log_info(
            logger,
            "ingestion.worker.start",
            domain="orderbook",
            symbol=symbol,
            source_type=type(source).__name__,
            has_local_data=has_local_data,
            start_ts=START_TS,
            end_ts=END_TS,
        )
        ingestion_tasks.append(asyncio.create_task(worker.run(emit=emit_to_queue)))

    # -------------------------
    # Option chain ingestion
    # -------------------------
    for asset, handler in engine.option_chain_handlers.items():
        has_local_data = _has_option_chain_data(
            DATA_ROOT / "raw" / "option_chain",
            asset=asset,
            interval="1m",
        )
        if not has_local_data:
            log_warn(
                logger,
                "ingestion.worker.skipped_no_data",
                domain="option_chain",
                symbol=asset,
                root=str(DATA_ROOT / "raw" / "option_chain"),
            )
            continue
        source = OptionChainFileSource(
            root=DATA_ROOT / "raw" / "option_chain",
            asset=asset,
            interval="1m",
            start_ts=START_TS,
            end_ts=END_TS,
        )
        normalizer = DeribitOptionChainNormalizer(symbol=asset)
        worker = OptionChainWorker(
            source=source,
            normalizer=normalizer,
            symbol=asset,
            poll_interval=None,  # backtest: do not throttle
        )
        log_info(
            logger,
            "ingestion.worker.start",
            domain="option_chain",
            symbol=asset,
            source_type=type(source).__name__,
            has_local_data=has_local_data,
            start_ts=START_TS,
            end_ts=END_TS,
        )
        ingestion_tasks.append(asyncio.create_task(worker.run(emit=emit_to_queue)))

    # -------------------------
    # Sentiment ingestion
    # -------------------------
    for src, handler in engine.sentiment_handlers.items():
        has_local_data = _has_sentiment_data(
            DATA_ROOT / "raw" / "sentiment",
            provider=src,
        )
        if not has_local_data:
            log_warn(
                logger,
                "ingestion.worker.skipped_no_data",
                domain="sentiment",
                symbol=src,
                root=str(DATA_ROOT / "raw" / "sentiment"),
            )
            continue
        source = SentimentFileSource(
            root=DATA_ROOT / "raw" / "sentiment",
            provider=src,
            start_ts=START_TS,
            end_ts=END_TS,
        )
        normalizer = SentimentNormalizer(symbol=src, provider=src)
        worker = SentimentWorker(
            source=source,
            normalizer=normalizer,
            # provider=src,
            poll_interval=None,  # backtest: do not throttle
        )
        log_info(
            logger,
            "ingestion.worker.start",
            domain="sentiment",
            symbol=src,
            source_type=type(source).__name__,
            has_local_data=has_local_data,
            start_ts=START_TS,
            end_ts=END_TS,
        )
        ingestion_tasks.append(asyncio.create_task(worker.run(emit=emit_to_queue)))

    # -------------------------------------------------
    # 3. Run deterministic backtest (time only)
    # -------------------------------------------------
    driver = BacktestDriver(
        engine=engine,
        spec=engine.spec,
        start_ts=START_TS,
        end_ts=END_TS,
        tick_queue=tick_queue,
    )

    log_info(logger, "app.backtest.start", start_ts=START_TS, end_ts=END_TS)
    await driver.run()

    log_info(logger, "ingestion.worker.stop", count=len(ingestion_tasks))
    for t in ingestion_tasks:
        t.cancel()
    if ingestion_tasks:
        await asyncio.gather(*ingestion_tasks, return_exceptions=True)

    # -------------------------------------------------
    # 4. Final snapshot / reports
    # -------------------------------------------------
    log_info(logger, "app.backtest.done")
    log_info(logger, "app.backtest.final_portfolio", portfolio=engine.portfolio.state().to_dict())


if __name__ == "__main__":
    asyncio.run(main())
