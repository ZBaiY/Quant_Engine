# features/loader.py

from quant_engine.features.registry import build_feature
from quant_engine.features.extractor import FeatureExtractor
from quant_engine.utils.logger import get_logger, log_debug


class FeatureLoader:
    _logger = get_logger(__name__)

    @staticmethod
    def from_config(cfg: dict, data_handler):
        """
        cfg example:
        {
            "features": [
                {"type": "RSI", "params": {"period": 14}},
                {"type": "MACD", "params": {"fast": 12, "slow": 26}},
                {"type": "VOLATILITY", "params": {"window": 30}}
            ]
        }
        """

        log_debug(FeatureLoader._logger, "FeatureLoader received config", config=cfg)
        log_debug(FeatureLoader._logger, "FeatureLoader building FeatureExtractor", feature_keys=[f["type"] for f in cfg["features"]])

        return FeatureExtractor(
            historical=data_handler.historical,
            realtime=data_handler.realtime,
            feature_config={f["type"]: f.get("params", {}) for f in cfg["features"]}
        )