# contracts/portfolio.py
from typing import Protocol, Dict
from dataclasses import dataclass


@dataclass
class PositionRecord:
    qty: float
    entry_price: float
    unrealized_pnl: float = 0.0


@dataclass
class PortfolioState:
    snapshot_dict: Dict

    def snapshot(self) -> Dict:
        return self.snapshot_dict


class PortfolioManagerProto(Protocol):
    """
    Core accounting interface.
    Receives fills from MatchingEngine.
    Updates positions, PnL, metrics.
    """
    def apply_fill(self, fill: Dict):
        """Update portfolio based on fill dict."""
        ...

    def state(self) -> PortfolioState:
        """Return current state."""
        ...