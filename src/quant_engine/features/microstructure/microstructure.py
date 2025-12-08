# src/quant_engine/features/microstructure/microstructure.py
# Microstructure feature channels (L2/L3 orderbook derived)
from quant_engine.contracts.feature import FeatureChannel
from ..registry import register_feature
from quant_engine.utils.logger import get_logger, log_debug


@register_feature("SPREAD")
class SpreadFeature(FeatureChannel):
    """Best bid/ask spread."""
    _logger = get_logger(__name__)
    def __init__(self, symbol=None, **kwargs):
        self._symbol = symbol
        self._value = None

    @property
    def symbol(self):
        return self._symbol

    def initialize(self, context):
        ob = context["orderbook_realtime"]
        snap = ob.latest_snapshot()
        self._value = float(snap.best_ask - snap.best_bid)

    def update(self, context):
        ob = context["orderbook_realtime"]
        snap = ob.latest_snapshot()
        self._value = float(snap.best_ask - snap.best_bid)

    def output(self):
        assert self._value is not None, "SpreadFeature.output() called before initialize()"
        return {"spread": self._value}


@register_feature("IMBALANCE")
class OrderImbalanceFeature(FeatureChannel):
    """Orderbook imbalance = (bid_size - ask_size) / (sum)."""
    _logger = get_logger(__name__)
    def __init__(self, symbol=None, **kwargs):
        self._symbol = symbol
        self._value = None

    @property
    def symbol(self):
        return self._symbol

    def initialize(self, context):
        ob = context["orderbook_realtime"]
        snap = ob.latest_snapshot()
        num = snap.best_bid_size - snap.best_ask_size
        den = snap.best_bid_size + snap.best_ask_size + 1e-9
        self._value = float(num / den)

    def update(self, context):
        ob = context["orderbook_realtime"]
        snap = ob.latest_snapshot()
        num = snap.best_bid_size - snap.best_ask_size
        den = snap.best_bid_size + snap.best_ask_size + 1e-9
        self._value = float(num / den)

    def output(self):
        assert self._value is not None, "OrderImbalanceFeature.output() called before initialize()"
        return {"order_imbalance": self._value}