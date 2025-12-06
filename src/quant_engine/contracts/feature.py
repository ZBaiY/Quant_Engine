from typing import Protocol, Dict, Any

class FeatureChannel(Protocol):
    symbol: str | None

    def __init__(self, symbol=None, **kwargs):
        self.symbol = symbol
        
    def compute(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        context = {
            "ohlcv": DataFrame,
            "option_chain_handler": OptionChainDataHandler,
            "sentiment_loader": SentimentLoader,
            "orderbook": OrderbookDataHandler,
            ...
        }
        """
        ...