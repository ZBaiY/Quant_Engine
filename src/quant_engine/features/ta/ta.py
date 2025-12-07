# features/ta.py
from quant_engine.contracts.feature import FeatureChannel
from ..registry import register_feature
import pandas as pd
from quant_engine.utils.logger import get_logger, log_debug

@register_feature("RSI")
class RSIFeature(FeatureChannel):
    _logger = get_logger(__name__)
    def __init__(self, symbol=None, **kwargs):
        self._symbol = symbol
        self.period = kwargs.get("period", 14)
        self._rsi = None
        self._avg_up = None
        self._avg_down = None

    @property
    def symbol(self):
        return self._symbol

    def required_window(self) -> int:
        return self.period + 1

    def initialize(self, context):
        data = context["ohlcv"]
        delta = data["close"].diff()
        up = delta.clip(lower=0)
        down = (-delta.clip(upper=0))

        # Wilder's smoothing initialization
        self._avg_up = up.rolling(self.period).mean().iloc[-1]
        self._avg_down = down.rolling(self.period).mean().iloc[-1]

        rs = self._avg_up / (self._avg_down + 1e-12)
        self._rsi = float(rs)

    def update(self, context):
        bar = context["ohlcv"]
        close = bar["close"].iloc[0]
        prev_close = context["realtime"].prev_close(self._symbol) if hasattr(context["realtime"], "prev_close") else None

        if prev_close is None:
            return

        delta = close - prev_close
        up = max(delta, 0)
        down = max(-delta, 0)

        # Wilder incremental smoothing
        self._avg_up = (self._avg_up * (self.period - 1) + up) / self.period
        self._avg_down = (self._avg_down * (self.period - 1) + down) / self.period

        rs = self._avg_up / (self._avg_down + 1e-12)
        self._rsi = float(rs)

    def output(self):
        assert self._rsi is not None, "RSIFeature.output() called before initialize()"
        return {"rsi": self._rsi}


@register_feature("MACD")
class MACDFeature(FeatureChannel):
    _logger = get_logger(__name__)
    def __init__(self, symbol=None, **kwargs):

        self._symbol = symbol
        self.fast = kwargs.get("fast", 12)
        self.slow = kwargs.get("slow", 26)
        self.signal = kwargs.get("signal", 9)
        self._ema_fast = None
        self._ema_slow = None
        self._macd = None

    @property
    def symbol(self):
        return self._symbol

    def required_window(self) -> int:
        return self.slow + 1

    def initialize(self, context):
        data = context["ohlcv"]
        close = data["close"]

        self._ema_fast = close.ewm(span=self.fast, adjust=False).mean().iloc[-1]
        self._ema_slow = close.ewm(span=self.slow, adjust=False).mean().iloc[-1]
        self._macd = float(self._ema_fast - self._ema_slow)

    def update(self, context):
        bar = context["ohlcv"]
        close = bar["close"].iloc[0]

        k_fast = 2 / (self.fast + 1)
        k_slow = 2 / (self.slow + 1)

        self._ema_fast = (close - self._ema_fast) * k_fast + self._ema_fast
        self._ema_slow = (close - self._ema_slow) * k_slow + self._ema_slow
        self._macd = float(self._ema_fast - self._ema_slow)

    def output(self):
        assert self._macd is not None, "MACDFeature.output() called before initialize()"
        return {"macd": self._macd}

@register_feature("ADX")
class ADXFeature(FeatureChannel):
    _logger = get_logger(__name__)

    def __init__(self, symbol=None, **kwargs):
        self._symbol = symbol
        self.period = kwargs.get("period", 14)

        self._tr = None
        self._dm_pos = None
        self._dm_neg = None

        self._atr = None
        self._di_pos = None
        self._di_neg = None
        self._adx = None

    @property
    def symbol(self):
        return self._symbol

    def required_window(self) -> int:
        return self.period + 1

    def initialize(self, context):
        data = context["ohlcv"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)

        tr = (
            pd.concat(
                [
                    (high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
        )

        up_move = high.diff()
        down_move = -low.diff()

        dm_pos = ((up_move > down_move) & (up_move > 0)) * up_move
        dm_neg = ((down_move > up_move) & (down_move > 0)) * down_move

        self._tr = tr.rolling(self.period).sum().iloc[-1]
        self._dm_pos = dm_pos.rolling(self.period).sum().iloc[-1]
        self._dm_neg = dm_neg.rolling(self.period).sum().iloc[-1]

        self._di_pos = 100 * (self._dm_pos / (self._tr + 1e-12))
        self._di_neg = 100 * (self._dm_neg / (self._tr + 1e-12))

        dx = 100 * (abs(self._di_pos - self._di_neg) /
                    (self._di_pos + self._di_neg + 1e-12))

        self._adx = float(dx)

    def update(self, context):
        bar = context["ohlcv"]
        high = bar["high"].iloc[0]
        low = bar["low"].iloc[0]
        close = bar["close"].iloc[0]

        prev_close = context["realtime"].prev_close(self._symbol)
        prev_high = context["realtime"].window_df(2)["high"].iloc[-2]
        prev_low = context["realtime"].window_df(2)["low"].iloc[-2]

        tr_ = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )

        up_move = high - prev_high
        down_move = prev_low - low

        dm_pos_ = up_move if (up_move > down_move and up_move > 0) else 0
        dm_neg_ = down_move if (down_move > up_move and down_move > 0) else 0

        self._tr = self._tr - (self._tr / self.period) + tr_
        self._dm_pos = self._dm_pos - (self._dm_pos / self.period) + dm_pos_
        self._dm_neg = self._dm_neg - (self._dm_neg / self.period) + dm_neg_

        self._di_pos = 100 * (self._dm_pos / (self._tr + 1e-12))
        self._di_neg = 100 * (self._dm_neg / (self._tr + 1e-12))

        dx = 100 * (abs(self._di_pos - self._di_neg) /
                    (self._di_pos + self._di_neg + 1e-12))

        self._adx = float(dx)

    def output(self):
        assert self._adx is not None, "ADXFeature.output() called before initialize()"
        return {"adx": self._adx}