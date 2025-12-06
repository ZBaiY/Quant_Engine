from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    side: str                 # "BUY" or "SELL"
    qty: float
    order_type: str           # "MARKET" or "LIMIT"
    price: Optional[float]    # None for MARKET orders
    timestamp: Optional[float] = None
    tag: str = ""             # strategy tag

    def to_dict(self):
        return {
            "side": self.side,
            "qty": self.qty,
            "order_type": self.order_type,
            "price": self.price,
            "timestamp": self.timestamp,
            "tag": self.tag,
        }