from __future__ import annotations
from typing import Set, Dict, Any, List

from quant_engine.utils.logger import get_logger, log_debug

_logger = get_logger(__name__)


def discover_symbols(
    primary: str,
    features_user: List[Dict[str, Any]],
    model_cfg: Dict[str, Any],
) -> Set[str]:
    """
 
    Collect all symbols required by:
        - primary symbol
        - user features (feature["symbol"])
        - user feature params (e.g., {"ref": "ETHUSDT"})
        - model secondary symbol (e.g., {"params": {"secondary": "ETHUSDT"}})

    Returns a deduplicated symbol set.
    """

    log_debug(_logger, "Discovering symbols", primary=primary)

    symbols: Set[str] = set()
    symbols.add(primary)

    # ---- From user features ----
    for item in features_user:
        sym = item.get("symbol")
        if sym:
            symbols.add(sym)

        params = item.get("params", {})
        ref = params.get("ref")
        if ref:
            symbols.add(ref)

    # ---- From model config ----
    model_params = model_cfg.get("params", {})

    # common name: secondary (pair trading)
    if "secondary" in model_params:
        symbols.add(model_params["secondary"])

    # general: if model has any ref symbol
    if "ref" in model_params:
        symbols.add(model_params["ref"])

    log_debug(_logger, "Symbols resolved", symbols=list(symbols))
    return symbols
