# features/extractor.py
from quant_engine.contracts.feature import FeatureChannel
from quant_engine.data.historical import HistoricalDataHandler
from quant_engine.data.realtime import RealTimeDataHandler
from .registry import build_feature
from quant_engine.utils.logger import get_logger, log_debug, log_info

"""
extractor = FeatureExtractor(
    historical=hist,
    realtime=rt,
    feature_names=["RSI", "MACD"],
    feature_kwargs={
        "RSI": {"period": 14},
        "MACD": {"fast": 12, "slow": 26},
    }
)
features = extractor.compute()
"""

class FeatureExtractor:
    _logger = get_logger(__name__)
    def __init__(
        self,
        historical: HistoricalDataHandler,
        realtime: RealTimeDataHandler,
        feature_config: dict[str, dict],
    ):
        """
        feature_config example:
        {
            "RSI": {"period": 14},
            "MACD": {"fast": 12, "slow": 26},
            "SPREAD": {},
            "REALIZED_VOL": {"window": 30}
        }
        """
        self.historical = historical
        self.realtime = realtime
        log_debug(self._logger, "FeatureExtractor initialized", feature_config=feature_config)

        self.channels: list[FeatureChannel] = [
            build_feature(name, **params)
            for name, params in feature_config.items()
        ]

    def compute(self) -> dict:
        log_debug(self._logger, "FeatureExtractor compute() called")
        df = self.realtime.window_df()
        log_debug(self._logger, "FeatureExtractor received dataframe", rows=len(df))
        result = {}
        for ch in self.channels:
            result.update(ch.compute(df))
        log_debug(self._logger, "FeatureExtractor computed features", feature_keys=list(result.keys()))
        return result