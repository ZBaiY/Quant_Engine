# models/momentum.py
from quant_engine.contracts.model import ModelBase, parse_feature_name
from .registry import register_model


def _find_feature_name(
    required_features: set[str],
    *,
    ftype: str,
    purpose: str,
    symbol: str,
    prefer_last: bool,
) -> str | None:
    match: str | None = None
    for name in required_features:
        try:
            t, p, s, _r = parse_feature_name(name)
        except Exception:
            continue
        if t == ftype and p == purpose and s == symbol:
            match = name
            if not prefer_last:
                break
    return match


@register_model("RSI-MODEL")
class RSIMomentumModel(ModelBase):
    # design-time capability requirement
    required_feature_types = {"RSI"}

    def __init__(self, symbol: str, **kwargs):
        overbought = int(kwargs.get("overbought", 70))
        oversold = int(kwargs.get("oversold", 30))
        super().__init__(symbol=symbol, **kwargs)
        self.ob = float(overbought)
        self.os = float(oversold)

    def predict(self, features: dict) -> float:
        # Prefer semantic lookup via bound feature index
        try:
            rsi = float(self.fget(features, ftype="RSI", purpose="MODEL"))
        except Exception:
            # Fallback: locate RSI feature in validation names
            rsi_name = _find_feature_name(
                getattr(self, "required_features", set()),
                ftype="RSI",
                purpose="MODEL",
                symbol=self.symbol,
                prefer_last=False,
            )
            if rsi_name is None:
                return 0.0
            rsi = float(features.get(rsi_name, 0.0))

        if rsi > self.ob:
            return -1.0   # mean reversion short
        if rsi < self.os:
            return 1.0    # mean reversion long
        return 0.0
    
@register_model("MACD-MODEL")
class MACDMomentumModel(ModelBase):
    # design-time capability requirements
    required_feature_types = {"MACD", "MACD_SIGNAL"}

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol=symbol, **kwargs)

    def predict(self, features: dict) -> float:
        # Prefer semantic lookup via bound feature index
        try:
            macd = float(self.fget(features, ftype="MACD", purpose="MODEL"))
            signal = float(self.fget(features, ftype="MACD_SIGNAL", purpose="MODEL"))
        except Exception:
            # Fallback: locate by parsing validation names (order-independent)
            required = getattr(self, "required_features", set())
            macd_name = _find_feature_name(
                required,
                ftype="MACD",
                purpose="MODEL",
                symbol=self.symbol,
                prefer_last=True,
            )
            sig_name = _find_feature_name(
                required,
                ftype="MACD_SIGNAL",
                purpose="MODEL",
                symbol=self.symbol,
                prefer_last=True,
            )
            if macd_name is None or sig_name is None:
                return 0.0
            macd = float(features.get(macd_name, 0.0))
            signal = float(features.get(sig_name, 0.0))

        return 1.0 if macd > signal else -1.0
