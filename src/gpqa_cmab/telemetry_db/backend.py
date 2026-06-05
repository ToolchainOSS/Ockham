"""Backend abstraction for the telemetry event store.

The :class:`TelemetryBackend` Protocol decouples the recorder from any
concrete database driver. SQLite and Postgres each provide an implementation
that satisfies this Protocol; nothing else in the codebase imports a driver
directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Literal, Protocol, runtime_checkable

from gpqa_cmab.telemetry_db.schema import EventType, TelemetryEvent

BackendKind = Literal["sqlite", "postgres"]


@runtime_checkable
class TelemetryBackend(Protocol):
    """Minimal, idempotent, append-only event store.

    All implementations MUST guarantee:

    * ``append`` is *durably committed* by the time it returns (no implicit
      buffering that can be lost on crash).
    * Re-inserting the same ``event_uuid`` is a silent no-op (idempotent
      writes — safe to retry on transient failure).
    * ``iter_events`` returns rows sorted by ``(ts_utc, id)`` ascending.
    """

    kind: BackendKind

    def initialize(self) -> None:
        """Create tables/indexes if missing. Safe to call repeatedly."""

    def append(self, event: TelemetryEvent) -> None:
        """Durably append a single event. Idempotent on ``event_uuid``."""

    def append_many(self, events: Sequence[TelemetryEvent]) -> int:
        """Durably append a batch in a single transaction. Returns the
        number of rows actually inserted (duplicates are skipped)."""

    def iter_events(
        self,
        *,
        run_id: str | None = None,
        event_types: Iterable[EventType] | None = None,
        since_utc: str | None = None,
        until_utc: str | None = None,
        limit: int | None = None,
    ) -> Iterator[TelemetryEvent]:
        """Stream events matching the filter, oldest-first."""

    def count_events(self, *, run_id: str | None = None) -> int:
        """Cheap row-count, optionally scoped to a single run."""

    def close(self) -> None:
        """Release the underlying connection. Safe to call repeatedly."""


__all__ = ["BackendKind", "TelemetryBackend"]
