# quant_engine/data/derivatives/loader.py
import pandas as pd
from .option_contract import OptionContract, OptionType
from .option_chain import OptionChain
import numpy as np

def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value

class OptionChainLoader:
    """
    Loads raw option chain data from CSV / Parquet / API.
    Produces OptionChain objects suitable for FeatureLayer or IV modeling layer.
    """

    def load_from_csv(self, path: str) -> list[OptionChain]:
        df = pd.read_csv(path)
        return self._group_by_expiry(df)

    def load_from_dataframe(self, df: pd.DataFrame) -> list[OptionChain]:
        return self._group_by_expiry(df)

    def _group_by_expiry(self, df: pd.DataFrame) -> list[OptionChain]:
        """
        Expect df to have columns:
        ['symbol', 'expiry', 'strike', 'type', 'bid', 'ask', 'last',
         'volume', 'oi', 'iv', 'delta', 'gamma', 'vega', 'theta']
        """
        chains = []
        for expiry, group in df.groupby("expiry"):
            expiry = str(expiry)
            symbol = group['symbol'].iloc[0]

            contracts = []
            for _, row in group.iterrows():
                contracts.append(
                    OptionContract(
                        symbol=symbol,
                        expiry=expiry,
                        strike=row['strike'],
                        option_type=OptionType(row['type']),
                        bid=_clean(row.get('bid')),
                        ask=_clean(row.get('ask')),
                        last=_clean(row.get('last')),
                        volume=_clean(row.get('volume')),
                        open_interest=_clean(row.get('oi')),
                        implied_vol=_clean(row.get('iv')),
                        delta=_clean(row.get('delta')),
                        gamma=_clean(row.get('gamma')),
                        vega=_clean(row.get('vega')),
                        theta=_clean(row.get('theta')),
                    )
                )

            chains.append(OptionChain(symbol=symbol, expiry=expiry, contracts=contracts))

        return chains