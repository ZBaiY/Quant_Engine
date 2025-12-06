from __future__ import annotations
from typing import Optional, Dict, Any, List

from quant_engine.contracts.feature import FeatureChannel
from quant_engine.data.ohlcv.historical import HistoricalDataHandler
from quant_engine.data.ohlcv.realtime import RealTimeDataHandler
from quant_engine.data.derivatives.chain_handler import OptionChainDataHandler
from quant_engine.sentiment.loader import SentimentLoader
from .registry import build_feature

from quant_engine.utils.logger import get_logger, log_debug


class FeatureExtractor:
    """
    Unified feature extractor — multi-source and multi-symbol ready.

    Context passed to each FeatureChannel:
    {
        "ohlcv": df or dict[symbol -> df],
        "historical": HistoricalDataHandler,
        "realtime": RealTimeDataHandler,
        "option_chain": OptionChainDataHandler | None,
        "sentiment": SentimentLoader | None,
    }

    Each FeatureChannel: compute(context) -> dict[str, float]
    """

    _logger = get_logger(__name__)

    def __init__(
        self,
        historical_ohlcv: HistoricalDataHandler,
        realtime_ohlcv: RealTimeDataHandler,
        option_chain_handler: Optional[OptionChainDataHandler] = None,
        sentiment_loader: Optional[SentimentLoader] = None,
        feature_config: List[Dict[str, Any]] | None = None,
    ):
        log_debug(self._logger, "Initializing FeatureExtractor")

        self.historical_ohlcv = historical_ohlcv
        self.realtime_ohlcv = realtime_ohlcv
        self.option_chain_handler = option_chain_handler
        self.sentiment_loader = sentiment_loader

        feature_config = feature_config or []

        self.channels = [
            build_feature(
                item["type"],
                symbol=item.get("symbol"),
                **item.get("params", {})
            )
            for item in feature_config
        ]

        log_debug(
            self._logger,
            "FeatureExtractor channels loaded",
            channels=[type(c).__name__ for c in self.channels]
        )

    # ----------------------------------------------------------------------
    # Main compute()
    # ----------------------------------------------------------------------
    def compute(self) -> Dict[str, Any]:
        """
        Main interface — compute all features from all channels.
        """
        #   - if realtime returns dict[symbol → df], context 100% supports it
        #   - if single df, same code works
        ohlcv_window = self.realtime_ohlcv.window_df()

        context = {
            "ohlcv": ohlcv_window,
            "historical": self.historical_ohlcv,
            "realtime": self.realtime_ohlcv,
            "option_chain": self.option_chain_handler,
            "sentiment": self.sentiment_loader,
        }

        result: Dict[str, Any] = {}

        for ch in self.channels:
            features = ch.compute(context)
            if not isinstance(features, dict):
                raise TypeError(
                    f"FeatureChannel {type(ch).__name__}.compute() must return dict"
                )
            result.update(features)

        log_debug(self._logger, "FeatureExtractor computed features", keys=list(result.keys()))
        return result