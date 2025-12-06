# quant_engine/features/iv_surface.py

from quant_engine.contracts.feature import FeatureChannel
from quant_engine.data.derivatives.option_chain import OptionChain
from quant_engine.iv.surface import IVSurfaceModel  # future SABR/SSVI model
from quant_engine.iv.ssvi import SSVIModel          # optional
from quant_engine.iv.sabr import SABRModel          # optional

from quant_engine.utils.logger import get_logger, log_debug, log_info

from quant_engine.features.registry import register_feature
from quant_engine.data.derivatives.chain_handler import OptionChainDataHandler

"""
IVSurfaceFeature
----------------------------------------
FeatureChannel that consumes OptionChain and produces
surface-based features such as:
    - atm_iv
    - skew_25d
    - curvature
    - ssvi_eta
    - sabr_alpha

This feature does NOT do SABR/SSVI implementation itself.
It only delegates to iv/surface.py which does the heavy math.
"""

@register_feature("IV_SURFACE")
class IVSurfaceFeature(FeatureChannel):
    _logger = get_logger(__name__)
    def __init__(
        self,
        symbol=None,
        **kwargs
    ):
        
        self.symbol = symbol
        self.expiry = kwargs.get("expiry", "next")
        self.model = kwargs.get("model", "SSVI")
        self.model_kwargs = kwargs.get("model_kwargs", {}) or {}
        log_debug(
            self._logger,
            "IVSurfaceFeature initialized",
            symbol=self.symbol,
            expiry=self.expiry,
            model=self.model,
            model_kwargs=self.model_kwargs,
        )

    def _fit_surface(self, chain: OptionChain):
        """
        Delegates surface fitting to modeling layer.
        """
        log_debug(self._logger, "IVSurfaceFeature fitting surface", model=self.model)
        match self.model.upper():
            case "SSVI":
                model = SSVIModel(**self.model_kwargs)
                return model.fit(chain)

            case "SABR":
                model = SABRModel(**self.model_kwargs)
                return model.fit(chain)

            case _:
                raise ValueError(f"Unsupported IV model: {self.model}")

    def compute(self, context) -> dict:
        # context is a dictionary containing runtime information; OHLCV component is ignored.
        """
        Feature extraction pipeline:
        1. Retrieve option chain handler from context
        2. Load option chain for given expiry
        3. Fit the SABR/SSVI surface
        4. Extract simple surface-based features
        """
        log_debug(self._logger, "IVSurfaceFeature compute() called")
        chain_handler = context.get("option_chain_handler")
        if chain_handler is None:
            log_debug(self._logger, "IVSurfaceFeature missing chain handler in context")
            return {
                "ivsurf_atm": None,
                "ivsurf_skew": None,
                "ivsurf_curvature": None,
            }
        chain = chain_handler.get_chain(self.expiry)
        if chain is None:
            log_debug(self._logger, "IVSurfaceFeature missing chain", expiry=self.expiry)
            return {
                "ivsurf_atm": None,
                "ivsurf_skew": None,
                "ivsurf_curvature": None,
            }

        surface = self._fit_surface(chain)

        # Skeleton-level feature extraction
        atm_iv = surface.atm_iv()
        skew = surface.smile_slope()
        curvature = surface.smile_curvature()

        log_debug(self._logger, "IVSurfaceFeature computed surface features", atm_iv=atm_iv, skew=skew, curvature=curvature)

        return {
            "ivsurf_atm": atm_iv,
            "ivsurf_skew": skew,
            "ivsurf_curvature": curvature,
        }