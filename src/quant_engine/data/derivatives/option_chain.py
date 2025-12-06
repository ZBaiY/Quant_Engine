from dataclasses import dataclass, field
from .option_contract import OptionContract, OptionType
from quant_engine.utils.logger import get_logger, log_debug

@dataclass
class OptionChain:
    _logger = get_logger(__name__)
    symbol: str               # BTCUSDT
    expiry: str               # '2025-06-27'
    contracts: list[OptionContract] = field(default_factory=list)

    def calls(self) -> list[OptionContract]:
        log_debug(self._logger, "OptionChain calls() queried", symbol=self.symbol, expiry=self.expiry)
        return [c for c in self.contracts if c.option_type == OptionType.CALL]

    def puts(self) -> list[OptionContract]:
        log_debug(self._logger, "OptionChain puts() queried", symbol=self.symbol, expiry=self.expiry)
        return [c for c in self.contracts if c.option_type == OptionType.PUT]

    def strikes(self) -> list[float]:
        log_debug(self._logger, "OptionChain strikes() queried", total_contracts=len(self.contracts))
        return sorted({c.strike for c in self.contracts})

    def get_contract(self, strike: float, callput: OptionType) -> OptionContract | None:
        log_debug(self._logger, "OptionChain get_contract() queried", strike=strike, callput=callput.value)
        for c in self.contracts:
            if c.strike == strike and c.option_type == callput:
                return c
        return None

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "expiry": self.expiry,
            "contracts": [c.to_dict() for c in self.contracts],
        }