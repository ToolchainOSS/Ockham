"""Postgres implementation of :class:`TelemetryBackend`.

Uses ``psycopg`` (v3) with autocommit transactions: each :meth:`append` call
is a one-statement transaction, durably fsync'd by Postgres before returning.
``psycopg`` is imported lazily so the base ``gpqa-cmab`` install stays free
of native dependencies — users only need ``pip install gpqa-cmab[telemetry-pg]``
when they want the Postgres backend.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from gpqa_cmab.telemetry_db.backend import BackendKind
from gpqa_cmab.telemetry_db.schema import EventType, TelemetryEvent, ddl_statements
from gpqa_cmab.telemetry_db.sqlite_backend import _build_select


class PostgresBackend:
    """Durable Postgres-backed event store.

    Each call opens a connection from the configured DSN; the cost is
    negligible compared to the durability guarantee, and it sidesteps
    long-lived connection / pool concerns. For high-throughput producers,
    pass a pre-built ``psycopg_pool.ConnectionPool`` via the ``pool``
    argument.
    """

    kind: BackendKind = "postgres"

    def __init__(self, dsn: str, *, pool: Any | None = None) -> None:
        self.dsn = dsn
        self._pool = pool
        # Lazy import so ``gpqa-cmab`` works without psycopg installed.
        try:
            import psycopg  # noqa: F401
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised in CI when extra missing
            raise ImportError(
                "Postgres telemetry backend requires the 'psycopg' package. "
                "Install with: pip install 'gpqa-cmab[telemetry-pg]'"
            ) from exc
        self._psycopg = __import__("psycopg")

    # -- lifecycle ------------------------------------------------------

    def _connect(self) -> Any:
        if self._pool is not None:
            return self._pool.connection()
        return self._psycopg.connect(self.dsn, autocommit=True)

    def initialize(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            for stmt in ddl_statements("postgres"):
                cur.execute(stmt)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()

    # -- writes ---------------------------------------------------------

    def append(self, event: TelemetryEvent) -> None:
        self.append_many([event])

    def append_many(self, events: Sequence[TelemetryEvent]) -> int:
        if not events:
            return 0
        rows = [_event_to_row_pg(ev) for ev in events]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                    INSERT INTO telemetry_events
                        (event_uuid, ts_utc, run_id, module, event_type,
                         schema_version, payload_json, parent_event_uuid, git_sha)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (event_uuid) DO NOTHING
                    """,
                rows,
            )
            return cur.rowcount or 0

    # -- reads ----------------------------------------------------------

    def iter_events(
        self,
        *,
        run_id: str | None = None,
        event_types: Iterable[EventType] | None = None,
        since_utc: str | None = None,
        until_utc: str | None = None,
        limit: int | None = None,
    ) -> Iterator[TelemetryEvent]:
        sql, params = _build_select(
            run_id=run_id,
            event_types=list(event_types) if event_types else None,
            since_utc=since_utc,
            until_utc=until_utc,
            limit=limit,
            placeholder="%s",
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            for row in cur:
                yield _row_to_event_pg(row)

    def count_events(self, *, run_id: str | None = None) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            if run_id is None:
                cur.execute("SELECT COUNT(*) FROM telemetry_events")
            else:
                cur.execute(
                    "SELECT COUNT(*) FROM telemetry_events WHERE run_id = %s",
                    (run_id,),
                )
            (count,) = cur.fetchone()
            return int(count)


# ---------------------------------------------------------------------------
# Marshalling — Postgres returns UUID and datetime natively, but we still
# send payload_json as a JSON-encoded string for parity with SQLite.
# ---------------------------------------------------------------------------


def _event_to_row_pg(event: TelemetryEvent) -> tuple[Any, ...]:
    return (
        str(event.event_uuid),
        event.ts_utc,
        event.run_id,
        event.module,
        event.event_type.value,
        event.schema_version,
        json.dumps(event.payload, sort_keys=True, default=str),
        str(event.parent_event_uuid) if event.parent_event_uuid else None,
        event.git_sha,
    )


def _row_to_event_pg(row: tuple[Any, ...]) -> TelemetryEvent:
    (
        _id,
        event_uuid,
        ts_utc,
        run_id,
        module,
        event_type,
        schema_version,
        payload,
        parent_event_uuid,
        git_sha,
    ) = row
    # psycopg returns JSONB as a parsed dict already
    if isinstance(payload, str):
        payload_dict = json.loads(payload) if payload else {}
    else:
        payload_dict = payload or {}
    return TelemetryEvent(
        event_uuid=event_uuid,
        ts_utc=ts_utc,
        run_id=run_id,
        module=module,
        event_type=EventType(event_type),
        schema_version=int(schema_version),
        payload=payload_dict,
        parent_event_uuid=parent_event_uuid,
        git_sha=git_sha,
    )


__all__ = ["PostgresBackend"]
