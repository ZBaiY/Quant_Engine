from __future__ import annotations
from typing import Dict, List, Set

from quant_engine.data.ohlcv.realtime import RealTimeDataHandler
from quant_engine.data.orderbook.realtime import RealTimeOrderbookHandler
from quant_engine.data.derivatives.chain_handler import OptionChainDataHandler
from quant_engine.sentiment.loader import SentimentLoader

from quant_engine.utils.logger import get_logger, log_debug

_logger = get_logger(__name__)


def build_multi_symbol_handlers(symbols: Set[str]) -> Dict[str, List]:
    """
    Construct multi-symbol handler groups for:
        - ohlcv
        - orderbook
        - option_chain
        - sentiment

    Each returns: dict[str → list[handlers]].

    This is the Version 3 data ingestion builder.
    """

    log_debug(_logger, "Building multi-symbol handlers", symbols=list(symbols))

    # ---- OHLCV (always required) ----
    ohlcv_handlers: List[RealTimeDataHandler] = [
        RealTimeDataHandler(symbol=s) for s in symbols
    ]

    # ---- Orderbook (optional, may be disabled per symbol) ----
    orderbook_handlers: List[RealTimeOrderbookHandler] = [
        RealTimeOrderbookHandler(symbol=s) for s in symbols
    ]

    # ---- Option chain (optional: some symbols may not have options) ----
    # For now, create handlers for all symbols (your IV module can skip missing ones)
    option_chain_handlers: List[OptionChainDataHandler] = [
        OptionChainDataHandler(symbol=s) for s in symbols
    ]

    # ---- Sentiment (optional, symbol-agnostic) ----
    # Most sentiment sources are per-symbol feeds, so one per symbol
    sentiment_handlers: List[SentimentLoader] = [
        SentimentLoader(symbol=s) for s in symbols
    ]

    handler_dict = {
        "ohlcv": ohlcv_handlers,
        "orderbook": orderbook_handlers,
        "option_chain": option_chain_handlers,
        "sentiment": sentiment_handlers,
    }

    log_debug(_logger, "Built handler groups",
              ohlcv=len(ohlcv_handlers),
              orderbook=len(orderbook_handlers),
              option_chain=len(option_chain_handlers),
              sentiment=len(sentiment_handlers))

    return handler_dict
