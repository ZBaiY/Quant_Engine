from __future__ import annotations
from typing import Protocol, Optional, Any
from quant_engine.data.derivatives.iv.surface import IVSurface


class IVSurfaceHandler(Protocol):
    """
    Protocol for IV Surface managers in TradeBot v4.

    This handler abstracts away the details of:
        - constructing IV surfaces (SSVI / SABR / spline / MC)
        - caching surfaces across ticks
        - serving features / models with stable IV queries

    All modes (backtest / mock / live) must satisfy this unified API.
    """

    # ---------------------------
    # Metadata
    # ---------------------------
    @property
    def symbol(self) -> str:
        """Return symbol whose option surface this handler manages."""
        ...

    # ---------------------------
    # Required core API
    # ---------------------------
    def latest_surface(self) -> Optional[IVSurface]:
        """
        [DEPRECATED — v4]

        Legacy pull-based API returning the last computed IV surface.
        Do NOT use in feature/model layers.

        Correct v4 access pattern:
            handler.get_snapshot(ts)

        Returns:
            The most recently built IV surface, or None.
        """
        ...

    def build_surface(self) -> Optional[IVSurface]:
        """
        [DEPRECATED — v4]

        Old push-based surface construction entrypoint.
        Engines should no longer call build_surface() directly.

        Correct v4 pattern:
            • on_tick(ts) updates internal state
            • get_snapshot(ts) returns aligned IVSurfaceSnapshot

        Returns:
            IVSurface or None.
        """
        ...

    def ready(self) -> bool:
        """
        Whether the handler has sufficient data (option chain snapshots)
        to construct a reliable IV surface.
        """
        ...

    # ---------------------------
    # Optional convenience API
    # ---------------------------
    def flush_cache(self) -> None:
        """Clear cached surfaces (used for backtest resets)."""
        return None

    def last_timestamp(self) -> Optional[int]:
        """Timestamp of the latest surface if available."""
        return None
