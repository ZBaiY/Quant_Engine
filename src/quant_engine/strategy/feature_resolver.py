from __future__ import annotations
from typing import List, Dict, Any, Set
from quant_engine.utils.logger import get_logger, log_debug
from quant_engine.models.registry import MODEL_REGISTRY
from quant_engine.risk.registry import RISK_REGISTRY

_logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Core features always injected for the primary symbol
# ----------------------------------------------------------------------
CORE_FEATURES = [
    {"type": "ATR"}
]


def _inject_core_features(primary_symbol: str) -> List[Dict[str, Any]]:
    """
    Core features required for every strategy:
        - ATR
        - VOLATILITY
    These always belong to the primary symbol.
    """
    return [
        {**f, "symbol": primary_symbol}
        for f in CORE_FEATURES
    ]


# ----------------------------------------------------------------------
# Merge + dedupe logic
# ----------------------------------------------------------------------
def _dedupe(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate by (type, symbol, sorted_params).
    """
    seen = set()
    out = []
    for f in features:
        key = (
            f["type"],
            f.get("symbol"),
            tuple(sorted(f.get("params", {}).items()))
        )
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


# ----------------------------------------------------------------------
# MAIN ENTRY: resolve_feature_config
# ----------------------------------------------------------------------
def resolve_feature_config(
    primary_symbol: str,
    user_features: List[Dict[str, Any]],
    model_required: List[str],
    model_secondary: List[str],
    risk_required: List[str],
) -> List[Dict[str, Any]]:
    """
    Version 4 resolver:
        1. Inject core features for primary symbol
        2. Merge user_features
        3. Inject model-required features (primary and secondary)
        4. Inject risk-required features
        5. Deduplicate
    """
    log_debug(_logger, "Resolving final feature config",
              primary=primary_symbol)

    # ---- 1. core features ----
    core_f = _inject_core_features(primary_symbol)
    log_debug(_logger, "Core features injected", count=len(core_f))

    # ---- 2. user features ----
    user_f = user_features or []
    log_debug(_logger, "User features loaded", count=len(user_f))

    # ---- 3. model-required (primary) ----
    model_f = [
        {"type": t.split("_")[0], "symbol": primary_symbol}
        for t in model_required
    ]

    # ---- 4. model-required (secondary / pair) ----
    secondary_f = []
    for t in model_secondary:
        base = t.split("_")[0]
        secondary_f.append({
            "type": base,
            "symbol": primary_symbol,
            "params": {"ref": None}  # resolver fills actual ref later
        })

    # ---- 5. risk-required features ----
    risk_f = [
        {"type": t.split("_")[0], "symbol": primary_symbol}
        for t in risk_required
    ]

    # ---- 6. merge & dedupe ----
    merged = core_f + user_f + model_f + secondary_f + risk_f
    final = _dedupe(merged)

    log_debug(_logger, "Final feature config resolved", count=len(final))
    return final
