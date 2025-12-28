from __future__ import annotations

from typing import Any, Mapping

from ingestion.contracts.tick import IngestionTick, Domain, normalize_tick, _coerce_epoch_ms
from ingestion.contracts.market import annotate_payload_market
from ingestion.contracts.normalize import Normalizer


class GenericOptionChainNormalizer(Normalizer):
    """
    Generic option chain normalizer.
    """
    
    symbol: str
    domain: Domain = "option_chain"
    venue: str
    asset_class: str
    currency: str | None
    calendar: str | None
    session: str | None
    timezone_name: str | None

    def __init__(
        self,
        symbol: str,
        *,
        venue: str = "deribit",
        asset_class: str = "option",
        currency: str | None = None,
        calendar: str | None = None,
        session: str | None = None,
        timezone_name: str | None = None,
    ):
        self.symbol = symbol
        self.venue = venue
        self.asset_class = asset_class
        self.currency = currency
        self.calendar = calendar
        self.session = session
        self.timezone_name = timezone_name

    def normalize(
        self,
        *,
        raw: Mapping[str, Any],
    ) -> IngestionTick:
        """
        Normalize a raw option chain payload into an IngestionTick.
        Expected (minimal) raw fields:
            - timestamp / ts / snapshot_time   (seconds or ms)
            - symbol / underlying
        """

        # --- timestamp extraction (best-effort, ingestion-level only) ---
        if "timestamp" in raw:
            ts = raw["timestamp"]
        elif "snapshot_time" in raw:
            ts = raw["snapshot_time"]
        elif "ts" in raw:
            ts = raw["ts"]
        else:
            raise ValueError("Option chain payload missing timestamp field")

        event_ts = _coerce_epoch_ms(ts)
        payload = annotate_payload_market(
            dict(raw),
            symbol=self.symbol,
            venue=self.venue,
            asset_class=self.asset_class,
            currency=self.currency,
            event_ts=event_ts,
            calendar=self.calendar,
            session=self.session,
            timezone_name=self.timezone_name,
        )

        return normalize_tick(
            timestamp=ts,
            domain=self.domain,
            symbol=self.symbol,
            payload=payload,
            data_ts=None,
        )
