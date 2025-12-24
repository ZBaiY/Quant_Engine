
from __future__ import annotations
from typing import Protocol, Callable, Awaitable
from ingestion.contracts.tick import IngestionTick


class IngestWorker(Protocol):
    """
    Ingestion worker contract.

    An IngestWorker is responsible ONLY for:
        - fetching / listening to external data sources
        - normalizing raw payloads
        - emitting IngestionTick objects

    It MUST NOT:
        - import quant_engine.*
        - know about Engine / Strategy / Mode
        - align data
        - cache windows
        - advance time
        - block on runtime semantics

    Lifecycle:
        world -> worker.run(emit_tick)
    """

    async def run(
        self,
        emit: Callable[[IngestionTick], None],
    ) -> Awaitable[None]:
        """
        Start the ingestion loop.

        Parameters
        ----------
        emit:
            Callback used to emit normalized IngestionTick objects downstream.
            Must be non-blocking and side-effect free from the worker's perspective.
        """
        ...