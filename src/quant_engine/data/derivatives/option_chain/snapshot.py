from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import datetime as dt
import re

import pandas as pd

from quant_engine.data.contracts.snapshot import Snapshot, MarketInfo, MarketSpec, ensure_market_spec
from quant_engine.utils.num import to_float


def to_ms_int(x: Any) -> int:
    """Coerce a timestamp-like value to epoch milliseconds as int."""
    return int(to_float(x))


# --- Deribit helpers (temporary fallback) ---
# Instrument example: BTC-28JUN24-60000-C
_DERIBIT_OPT_RE = re.compile(
    r"^(?P<underlying>[A-Z]+)-(?P<date>\d{2}[A-Z]{3}\d{2})-(?P<strike>[0-9.]+)-(?P<type>[CP])$"
)


def _parse_deribit_expiry_ts_ms(instrument_name: str) -> int:
    """Best-effort Deribit expiry timestamp (epoch ms, UTC).

    Prefer exchange-provided `expiration_timestamp` when available.
    This is only a fallback.
    """
    m = _DERIBIT_OPT_RE.match(instrument_name)
    if not m:
        raise ValueError(f"Unrecognized Deribit instrument_name: {instrument_name}")
    expiry_date = dt.datetime.strptime(m.group("date"), "%d%b%y").date()
    expiry_dt = dt.datetime(
        expiry_date.year,
        expiry_date.month,
        expiry_date.day,
        8,
        0,
        0,
        0,
        tzinfo=dt.timezone.utc,
    )
    return int(expiry_dt.timestamp() * 1000)


def _parse_deribit_cp(instrument_name: str) -> str | None:
    m = _DERIBIT_OPT_RE.match(instrument_name)
    if not m:
        return None
    t = m.group("type")
    return "C" if t == "C" else ("P" if t == "P" else None)


def _coerce_cp(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    u = s.upper()
    if u in {"C", "CALL"}:
        return "C"
    if u in {"P", "PUT"}:
        return "P"
    if u.startswith("CALL"):
        return "C"
    if u.startswith("PUT"):
        return "P"
    return None


# Any IV-ish column is treated as venue-fetched and kept only in aux.
_IV_KEY_RE = re.compile(r"^(iv)$|(_iv$)|(^iv_)|(_iv_)", re.IGNORECASE)


def _is_iv_col(c: str) -> bool:
    return bool(_IV_KEY_RE.search(c))


_GREEK_COLS = {
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "vomma",
    "volga",
    "charm",
    "speed",
    "zomma",
    "color",
}


# Minimal, IV-surface-relevant columns we keep in the main frame.
# Everything else goes into `aux` per row.
_SURFACE_KEEP_COLS = {
    "instrument_name",
    "expiry_ts",
    "strike",
    "cp",
    # optional: if you later join quotes/marks into the same frame
    "bid",
    "ask",
    "mid",
    "mark",
    "mark_price",
    "index_price",
    "underlying_price",
    "forward_price",
    "oi",
    "open_interest",
    "volume",
    # aux holder
    "aux",
}


@dataclass(frozen=True)
class OptionChainSnapshot(Snapshot):
    """Immutable option chain snapshot.

    Schema v2:
      - the chain payload is stored as a pandas DataFrame (`frame`).
      - the frame is *lean*: only IV-surface relevant columns are kept.
      - all other incoming columns are moved into a per-row dict column `aux`.
      - any fetched IV/greeks columns are treated as non-canonical and moved into aux as `*_fetch`.

    Canonical IV lives in iv_handler; greeks are computed in features.
    """

    data_ts: int
    symbol: str
    market: MarketSpec
    domain: str
    schema_version: int

    # normalized chain table
    frame: pd.DataFrame
    expiry_keys_ms: frozenset[int] | None = None
    term_keys_ms: dict[int, frozenset[int]] | None = None

    @staticmethod
    def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["instrument_name", "expiry_ts", "strike", "cp", "aux"])

        x = df.copy()

        # Map Deribit instrument metadata: expiration_timestamp -> expiry_ts
        if "expiry_ts" not in x.columns and "expiration_timestamp" in x.columns:
            x["expiry_ts"] = x["expiration_timestamp"]

        # Coerce expiry_ts (ms-int)
        if "expiry_ts" in x.columns:
            x["expiry_ts"] = pd.to_numeric(x["expiry_ts"], errors="coerce").fillna(0).astype("int64")

        # Coerce strike
        if "strike" in x.columns:
            x["strike"] = pd.to_numeric(x["strike"], errors="coerce")

        # Derive cp: option_type -> cp, else instrument_name fallback
        if "cp" not in x.columns:
            if "option_type" in x.columns:
                x["cp"] = x["option_type"].map(_coerce_cp)
            else:
                x["cp"] = None

        if "instrument_name" in x.columns:
            miss = x["cp"].isna() | (x["cp"].astype("string") == "")
            if bool(miss.any()):
                x.loc[miss, "cp"] = x.loc[miss, "instrument_name"].map(
                    lambda s: _parse_deribit_cp(str(s)) if s is not None else None
                )

        # Final fallback expiry_ts from instrument_name if still missing/zero
        if "expiry_ts" in x.columns and "instrument_name" in x.columns:
            miss_exp = (x["expiry_ts"].isna()) | (x["expiry_ts"].astype("int64") <= 0)
            if bool(miss_exp.any()):
                def _fallback_exp(v: Any) -> int | None:
                    try:
                        return _parse_deribit_expiry_ts_ms(str(v))
                    except Exception:
                        return None

                x.loc[miss_exp, "expiry_ts"] = x.loc[miss_exp, "instrument_name"].map(_fallback_exp)
                x["expiry_ts"] = pd.to_numeric(x["expiry_ts"], errors="coerce").fillna(0).astype("int64")

        # Ensure aux exists
        if "aux" not in x.columns:
            x["aux"] = [{} for _ in range(len(x))]
        else:
            # normalize aux values to dict
            x["aux"] = x["aux"].map(lambda v: dict(v) if isinstance(v, dict) else {})

        # Move fetched IV/greeks columns into aux as *_fetch (non-canonical)
        cols = list(x.columns)
        for c in cols:
            if not isinstance(c, str):
                continue
            lc = c.lower()
            if lc == "aux":
                continue
            if lc.endswith("_fetch") or _is_iv_col(c) or lc in _GREEK_COLS:
                def _move(v: Any, col: str = c) -> None:
                    # handled below via apply
                    return None

                # vectorized-ish: apply per row
                x["aux"] = x.apply(
                    lambda row, col=c: {**(row["aux"] if isinstance(row["aux"], dict) else {}), f"{col}_fetch" if not str(col).lower().endswith("_fetch") else str(col): row[col]},
                    axis=1,
                )
                x = x.drop(columns=[c])

        # Now move all non-surface columns into aux
        keep = set(_SURFACE_KEEP_COLS)
        for c in list(x.columns):
            if c in keep:
                continue
            # move into aux then drop
            x["aux"] = x.apply(
                lambda row, col=c: {**(row["aux"] if isinstance(row["aux"], dict) else {}), str(col): row[col]},
                axis=1,
            )
            x = x.drop(columns=[c])

        # Ensure required columns exist (even if None)
        for c in ("instrument_name", "expiry_ts", "strike", "cp"):
            if c not in x.columns:
                x[c] = None

        # Stable order
        order = [c for c in ("instrument_name", "expiry_ts", "strike", "cp", "aux") if c in x.columns]
        rest = [c for c in x.columns if c not in order]
        x = x[order + rest]

        # Sort for deterministic downstream processing
        try:
            x = x.sort_values(["expiry_ts", "strike", "cp", "instrument_name"], kind="stable")
        except Exception:
            pass

        return x.reset_index(drop=True)

    @classmethod
    def from_chain_aligned(
        cls,
        *,
        data_ts: int,
        symbol: str,
        market: MarketSpec | None = None,
        chain: pd.DataFrame,
        schema_version: int = 2,
        domain: str = "option_chain",
    ) -> "OptionChainSnapshot":
        dts = to_ms_int(data_ts)

        if not isinstance(chain, pd.DataFrame):
            raise TypeError("OptionChainSnapshot.from_chain_aligned expects `chain` as a pandas DataFrame")

        frame = cls._normalize_frame(chain)

        expiry_keys_ms: frozenset[int] | None
        if "expiry_ts" in frame.columns:
            try:
                xs = pd.to_numeric(frame["expiry_ts"], errors="coerce").dropna().astype("int64")
                expiry_keys_ms = frozenset(int(v) for v in xs.unique() if int(v) > 0)
            except Exception:
                expiry_keys_ms = frozenset()
        else:
            expiry_keys_ms = frozenset()

        return cls(
            data_ts=dts,
            symbol=symbol,
            market=ensure_market_spec(market),
            domain=domain,
            schema_version=int(schema_version),
            frame=frame,
            expiry_keys_ms=expiry_keys_ms,
            term_keys_ms=None,
        )

    @classmethod
    def from_chain(
        cls,
        ts: int,
        chain: pd.DataFrame,
        symbol: str,
        market: MarketSpec | None = None,
    ) -> "OptionChainSnapshot":
        return cls.from_chain_aligned(
            data_ts=ts,
            symbol=symbol,
            market=market,
            chain=chain,
        )

    def to_dict(self) -> Dict[str, Any]:
        assert isinstance(self.market, MarketInfo)
        return {
            "data_ts": self.data_ts,
            "symbol": self.symbol,
            "market": self.market.to_dict(),
            "domain": self.domain,
            "schema_version": self.schema_version,
            # store as records for JSON-compat
            "frame": self.frame,
        }
    def get_attr(self, key: str) -> Any:
        if not hasattr(self, key):
            raise AttributeError(f"{type(self).__name__} has no attribute {key!r}")
        return getattr(self, key)

    def get_expiry_keys_ms(self) -> frozenset[int]:
        cached = self.expiry_keys_ms
        if cached is not None:
            return cached
        frame = self.frame
        if frame is None or frame.empty or "expiry_ts" not in frame.columns:
            keys = frozenset()
        else:
            try:
                xs = frame["expiry_ts"].dropna().unique()
                keys = frozenset(int(v) for v in xs if v is not None and int(v) > 0)
            except Exception:
                keys = frozenset()
        object.__setattr__(self, "expiry_keys_ms", keys)
        return keys

    def get_term_keys_ms(self, term_bucket_ms: int) -> frozenset[int]:
        tb = int(term_bucket_ms)
        if tb <= 0:
            raise ValueError("term_bucket_ms must be > 0")
        cached = self.term_keys_ms
        if cached is None:
            cached = {}
        if tb in cached:
            return cached[tb]
        # term_keys_ms cached against immutable data_ts; data_ts must not change post creation.
        snap_ts = int(self.data_ts)
        keys = {
            (max(0, int(ex) - snap_ts) // tb) * tb
            for ex in self.get_expiry_keys_ms()
        }
        out = frozenset(int(k) for k in keys)
        cached[tb] = out
        if len(cached) > 4:
            cached.pop(next(iter(cached)))
        object.__setattr__(self, "term_keys_ms", cached)
        return out


def _empty_frame_like(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    try:
        return frame.iloc[0:0].copy()
    except Exception:
        return pd.DataFrame(columns=list(frame.columns) if hasattr(frame, "columns") else [])


def _slice_frame_for_expiry(frame: pd.DataFrame, *, expiry_ts: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_frame_like(frame)
    if "expiry_ts" not in frame.columns:
        return _empty_frame_like(frame)
    try:
        mask = pd.to_numeric(frame["expiry_ts"], errors="coerce").fillna(0).astype("int64") == int(expiry_ts)
        return frame.loc[mask].reset_index(drop=True)
    except Exception:
        return _empty_frame_like(frame)


def _slice_frame_for_term_bucket(
    frame: pd.DataFrame,
    *,
    snap_ts: int,
    term_key_ms: int,
    term_bucket_ms: int,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_frame_like(frame)
    if "expiry_ts" not in frame.columns:
        return _empty_frame_like(frame)
    try:
        expiry = pd.to_numeric(frame["expiry_ts"], errors="coerce").fillna(0).astype("int64")
        term = (expiry - int(snap_ts)).clip(lower=0)
        key = (term // int(term_bucket_ms)) * int(term_bucket_ms)
        mask = key == int(term_key_ms)
        return frame.loc[mask].reset_index(drop=True)
    except Exception:
        return _empty_frame_like(frame)


class OptionChainSnapshotView(OptionChainSnapshot):
    """Lazy view over an OptionChainSnapshot frame."""

    _base: OptionChainSnapshot
    _frame_filter: Callable[[pd.DataFrame], pd.DataFrame]
    _frame_cache: pd.DataFrame | None

    def __init__(self, *, base: OptionChainSnapshot, frame_filter: Callable[[pd.DataFrame], pd.DataFrame]):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_frame_filter", frame_filter)
        object.__setattr__(self, "_frame_cache", None)

        object.__setattr__(self, "data_ts", base.data_ts)
        object.__setattr__(self, "symbol", base.symbol)
        object.__setattr__(self, "market", base.market)
        object.__setattr__(self, "domain", base.domain)
        object.__setattr__(self, "schema_version", base.schema_version)

        # Preserve optional fields (e.g., arrival_ts) when present on the base snapshot.
        for k, v in getattr(base, "__dict__", {}).items():
            if k in {"data_ts", "symbol", "market", "domain", "schema_version", "frame"}:
                continue
            if k.startswith("_"):
                continue
            try:
                object.__setattr__(self, k, v)
            except Exception:
                pass

    @property
    def frame(self) -> pd.DataFrame:
        cached = getattr(self, "_frame_cache", None)
        if cached is None:
            try:
                view = self._frame_filter(self._base.frame)
            except Exception:
                view = self._base.frame
            view = view.copy()
            view["snapshot_data_ts"] = int(self._base.data_ts)
            arrival_ts = getattr(self._base, "arrival_ts", None)
            if arrival_ts is not None:
                view["snapshot_arrival_ts"] = int(arrival_ts)
            object.__setattr__(self, "_frame_cache", view)
            return view
        return cached

    @classmethod
    def for_expiry(cls, *, base: OptionChainSnapshot, expiry_ts: int) -> "OptionChainSnapshotView":
        ex = int(expiry_ts)
        if ex not in base.get_expiry_keys_ms():
            return cls(base=base, frame_filter=lambda f: _empty_frame_like(f))
        return cls(base=base, frame_filter=lambda f: _slice_frame_for_expiry(f, expiry_ts=ex))

    @classmethod
    def for_term_bucket(
        cls,
        *,
        base: OptionChainSnapshot,
        term_key_ms: int,
        term_bucket_ms: int,
    ) -> "OptionChainSnapshotView":
        snap_ts = int(base.data_ts)
        tk = int(term_key_ms)
        tb = int(term_bucket_ms)
        if tk not in base.get_term_keys_ms(tb):
            return cls(base=base, frame_filter=lambda f: _empty_frame_like(f))
        return cls(
            base=base,
            frame_filter=lambda f: _slice_frame_for_term_bucket(
                f,
                snap_ts=snap_ts,
                term_key_ms=tk,
                term_bucket_ms=tb,
            ),
        )
