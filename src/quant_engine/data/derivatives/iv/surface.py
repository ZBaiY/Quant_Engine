from abc import ABC, abstractmethod
from typing import Any, Iterable
from quant_engine.data.derivatives.option_chain.snapshot import OptionChainSnapshot


class IVSurface(ABC):
    """
    Abstract interface for implied volatility surfaces (SABR, SSVI, spline, MC, etc.)

    A valid surface must support:
        - Fitting from an OptionChainSnapshot
        - Querying σ(K, T)
        - ATM IV / skew / curvature summary stats
        - Expiry / strike grids for interpolation
        - Timestamp tracking
    """

    # --------------------------------------------------------
    # Core fitting API
    # --------------------------------------------------------
    @abstractmethod
    def fit(self, chain: OptionChainSnapshot):
        """
        Fit the IV surface from a frozen OptionChainSnapshot.
        Implementations must be deterministic and side-effect free.
        """
        raise NotImplementedError

    # --------------------------------------------------------
    # Core query API (must-have)
    # --------------------------------------------------------
    @abstractmethod
    def sigma(self, strike: float, expiry: str) -> float:
        """
        Return IV σ(K, T) for the given strike and expiry.
        Must be implemented by all surface models.
        """
        raise NotImplementedError

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------
    @abstractmethod
    def atm_iv(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def skew(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def curvature(self) -> float:
        raise NotImplementedError

    # --------------------------------------------------------
    # Grid helpers (optional but recommended)
    # --------------------------------------------------------
    def expiries(self) -> Iterable[str]:
        """Return available expiries used by this surface."""
        raise NotImplementedError

    def strikes(self) -> Iterable[float]:
        """Return available strikes used by this surface."""
        raise NotImplementedError

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------
    @abstractmethod
    def data_timestamp(self) -> float:
        """Observation timestamp of the chain used for fitting."""
        raise NotImplementedError

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary summarizing the surface (for logging/debugging)."""
        return {
            "atm_iv": self.atm_iv(),
            "skew": self.skew(),
            "curvature": self.curvature(),
        }