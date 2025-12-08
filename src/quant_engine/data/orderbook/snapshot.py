from dataclasses import dataclass, field
from typing import Optional, Dict, List
import pandas as pd


@dataclass
class OrderbookSnapshot:
    """
    Lightweight container for L1/L2 orderbook snapshots.

    This can be expanded later to full depth,
    but for now we store:
        - best bid/ask (L1)
        - optional aggregated depth (L2)
        - timestamp
    """

    timestamp: float
    symbol: str

    # L1
    best_bid: float
    best_bid_size: float
    best_ask: float
    best_ask_size: float

    # Optional L2 aggregated data
    bids: Optional[List[Dict[str, float]]] = field(default_factory=list)
    asks: Optional[List[Dict[str, float]]] = field(default_factory=list)

    def mid_price(self) -> float:
        """Return mid price if both bid and ask exist."""
        if self.best_bid is None or self.best_ask is None:
            return float("nan")
        return 0.5 * (self.best_bid + self.best_ask)

    def to_dict(self) -> Dict:
        """Convert to plain dict for logging or JSON."""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "best_bid": self.best_bid,
            "best_bid_size": self.best_bid_size,
            "best_ask": self.best_ask,
            "best_ask_size": self.best_ask_size,
            "bids": self.bids,
            "asks": self.asks,
            "mid": self.mid_price(),
        }

    @staticmethod
    def from_dataframe(df: pd.DataFrame, symbol: str) -> "OrderbookSnapshot":
        """
        Convert 1-row DataFrame into snapshot.
        Expected columns:
            timestamp, best_bid, best_bid_size, best_ask, best_ask_size
        """
        row = df.iloc[0]

        return OrderbookSnapshot(
            timestamp=float(row["timestamp"]),
            symbol=symbol,
            best_bid=float(row["best_bid"]),
            best_bid_size=float(row["best_bid_size"]),
            best_ask=float(row["best_ask"]),
            best_ask_size=float(row["best_ask_size"]),
        )
