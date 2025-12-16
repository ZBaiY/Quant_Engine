# quant_engine/data/sentiment/snapshot.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class SentimentSnapshot:
    timestamp: float          # observation timestamp (obs_ts)
    symbol: str
    source: str
    interval: str
    model: str
    score: float
    embedding: list[float] | None
    meta: dict[str, Any]
    latency: float            # engine_ts - obs_ts

    @classmethod
    def from_payload(
        cls,
        *,
        engine_ts: float,
        obs_ts: float,
        symbol: str,
        source: str,
        interval: str,
        model: str,
        score: float,
        embedding: list[float] | None,
        meta: dict[str, Any] | None = None,
    ) -> "SentimentSnapshot":
        return cls(
            timestamp=float(obs_ts),
            symbol=symbol,
            source=source,
            interval=interval,
            model=model,
            score=float(score),
            embedding=embedding,
            meta=dict(meta or {}),
            latency=float(engine_ts - obs_ts),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "source": self.source,
            "model": self.model,
            "score": self.score,
            "embedding": self.embedding,
            "meta": self.meta,
            "latency": self.latency,
        }
