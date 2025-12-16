"""
Strategy base (v4).

A Strategy is a declarative object that defines:
- required data domains
- default feature / model / decision / risk configuration

A Strategy does NOT:
- build infrastructure directly
- execute trading logic
- hold runtime state

Construction is delegated to StrategyLoader.
Execution is owned by StrategyEngine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Optional, Dict, Any, Type, TypeVar
from quant_engine.strategy.loader import StrategyLoader
import copy

T = TypeVar("T", bound="StrategyBase")

@dataclass
class StrategyBase:
    """
    Declarative strategy specification.

    Subclasses must declare:
        REQUIRED_DATA: set[str]

    Subclasses may declare default config blocks that will be merged
    into runtime strategy configs by StrategyLoader.
    """

    # Set by registry
    STRATEGY_NAME: str = "UNREGISTERED"

    # -----------------------------
    # Declarative contracts
    # -----------------------------

    # redundant, but making the class syntactically cleaner
    REQUIRED_DATA: Set[str] = field(default_factory=set) 

    # Concrete data specification (primary / secondary)
    DATA: Dict[str, Any] = field(default_factory=dict)

    # Feature specifications exactly as in cfg["features_user"]
    FEATURES_USER: list[Dict[str, Any]] = field(default_factory=list)

    # Default config blocks (optional)
    MODEL_CFG: Optional[Dict[str, Any]] = None
    DECISION_CFG: Optional[Dict[str, Any]] = None
    RISK_CFG: Optional[Dict[str, Any]] = None
    EXECUTION_CFG: Optional[Dict[str, Any]] = None
    PORTFOLIO_CFG: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    # -----------------------------
    # Validation
    # -----------------------------
    def validate(self) -> None:
        if not isinstance(self.REQUIRED_DATA, set):
            raise TypeError("REQUIRED_DATA must be a set[str]")
        if not all(isinstance(x, str) for x in self.REQUIRED_DATA):
            raise TypeError("REQUIRED_DATA must be a set[str]")
        if not self.REQUIRED_DATA:
            raise ValueError("REQUIRED_DATA must be non-empty")

        if not isinstance(self.DATA, dict):
            raise TypeError("DATA must be a dict")

        declared_domains: set[str] = set()

        primary = self.DATA.get("primary", {})
        if not isinstance(primary, dict):
            raise TypeError("DATA['primary'] must be a dict")

        for domain in primary.keys():
            declared_domains.add(domain)

        secondary = self.DATA.get("secondary", {})
        if secondary:
            if not isinstance(secondary, dict):
                raise TypeError("DATA['secondary'] must be a dict")
            for _, sec_block in secondary.items():
                if not isinstance(sec_block, dict):
                    raise TypeError("Each secondary symbol block must be a dict")
                for domain in sec_block.keys():
                    declared_domains.add(domain)

        missing = self.REQUIRED_DATA - declared_domains
        if missing:
            raise ValueError(
                f"Strategy '{self.STRATEGY_NAME}' requires data domains "
                f"{sorted(self.REQUIRED_DATA)}, but DATA only declares "
                f"{sorted(declared_domains)} (missing {sorted(missing)})"
            )

        if not isinstance(self.FEATURES_USER, list):
            raise TypeError("FEATURES_USER must be a list[dict]")
        for f in self.FEATURES_USER:
            if not isinstance(f, dict):
                raise TypeError("Each FEATURES_USER entry must be a dict")

        for name, block in [
            ("MODEL_CFG", self.MODEL_CFG),
            ("DECISION_CFG", self.DECISION_CFG),
            ("RISK_CFG", self.RISK_CFG),
            ("EXECUTION_CFG", self.EXECUTION_CFG),
            ("PORTFOLIO_CFG", self.PORTFOLIO_CFG),
        ]:
            if block is not None and not isinstance(block, dict):
                raise TypeError(f"{name} must be a dict or None")

    # =================================================================
    # Construction helpers (delegation, NOT ownership)
    # =================================================================

    def apply_defaults(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge Strategy-declared defaults into a runtime cfg dict.

        Explicit values in cfg always win.
        """
        merged = copy.deepcopy(cfg)

        if self.FEATURES_USER:
            merged.setdefault("features_user", self.FEATURES_USER)

        if self.MODEL_CFG:
            merged.setdefault("model", self.MODEL_CFG)

        if self.DECISION_CFG:
            merged.setdefault("decision", self.DECISION_CFG)

        if self.RISK_CFG:
            merged.setdefault("risk", self.RISK_CFG)

        if self.EXECUTION_CFG:
            merged.setdefault("execution", self.EXECUTION_CFG)

        if self.PORTFOLIO_CFG:
            merged.setdefault("portfolio", self.PORTFOLIO_CFG)

        if self.DATA:
            merged.setdefault("data", copy.deepcopy(self.DATA))

        return merged

    def to_dict(self) -> Dict[str, Any]:
        """
        Export this Strategy specification to a JSON‑serializable dict.
        """
        return {
            "strategy": {"name": self.STRATEGY_NAME},
            "required_data": sorted(self.REQUIRED_DATA),
            "data": copy.deepcopy(self.DATA),
            "features_user": copy.deepcopy(self.FEATURES_USER),
            "model": copy.deepcopy(self.MODEL_CFG),
            "decision": copy.deepcopy(self.DECISION_CFG),
            "risk": copy.deepcopy(self.RISK_CFG),
            "execution": copy.deepcopy(self.EXECUTION_CFG),
            "portfolio": copy.deepcopy(self.PORTFOLIO_CFG),
        }

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Construct a Strategy from a dict (e.g. JSON‑deserialized).
        """
        return cls(
            REQUIRED_DATA=set(data.get("required_data", [])),
            DATA=data.get("data", {}),
            FEATURES_USER=data.get("features_user", []),
            MODEL_CFG=data.get("model"),
            DECISION_CFG=data.get("decision"),
            RISK_CFG=data.get("risk"),
            EXECUTION_CFG=data.get("execution"),
            PORTFOLIO_CFG=data.get("portfolio"),
        )

    def build(self, mode, overrides: Dict[str, Any] | None = None):
        """Build a StrategyEngine using StrategyLoader.

        Args:
            mode: Engine mode (e.g., EngineMode.BACKTEST / REALTIME / MOCK).
            overrides: Runtime config overrides (does not mutate the Strategy).
        """
        if mode is None:
            raise ValueError("StrategyBase.build(mode=...) requires a non-None mode")

        # Local import to avoid import-time coupling/cycles.
   

        return StrategyLoader.from_config(
            strategy=self,
            mode=mode,
            overrides=overrides or {},
        )
