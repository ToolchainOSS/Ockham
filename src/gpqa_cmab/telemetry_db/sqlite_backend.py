"""SQLite implementation of :class:`TelemetryBackend`.

WAL journaling + ``synchronous=NORMAL`` give per-event durability without
fsync-on-every-write latency. Concurrent reads while a long experiment
writes (e.g. tailing logs) are unaffected by the writer.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from gpqa_cmab.telemetry_db.backend import BackendKind
from gpqa_cmab.telemetry_db.schema import EventType, TelemetryEvent, ddl_statements


class SqliteBackend:
    """Durable, single-process-friendly SQLite event store.

    Multiple processes can read concurrently (WAL); writers across processes
    are serialised by SQLite's BEGIN IMMEDIATE. Within one process the
    backend is thread-safe via an internal mutex.
    """

    kind: BackendKind = "sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle ------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self.path,
            isolation_level=None,  # autocommit mode, we manage transactions
            check_same_thread=False,
            timeout=30.0,
        )
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        self._conn = conn
        return conn

    def initialize(self) -> None:
        conn = self._connect()
        with self._lock:
            for stmt in ddl_statements("sqlite"):
                conn.execute(stmt)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # -- writes ---------------------------------------------------------

    def append(self, event: TelemetryEvent) -> None:
        self.append_many([event])

    def append_many(self, events: Sequence[TelemetryEvent]) -> int:
        if not events:
            return 0
        conn = self._connect()
        rows = [_event_to_row(ev) for ev in events]
        with self._lock:
            cur = conn.execute("BEGIN IMMEDIATE")
            try:
                cur.executemany(
                    """
                    INSERT OR IGNORE INTO telemetry_events
                        (event_uuid, ts_utc, run_id, module, event_type,
                         schema_version, payload_json, parent_event_uuid, git_sha)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                inserted = cur.rowcount
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            else:
                return inserted

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
            placeholder="?",
        )
        conn = self._connect()
        cursor = conn.execute(sql, params)
        try:
            for row in cursor:
                yield _row_to_event(row)
        finally:
            cursor.close()

    def count_events(self, *, run_id: str | None = None) -> int:
        conn = self._connect()
        if run_id is None:
            (count,) = conn.execute("SELECT COUNT(*) FROM telemetry_events").fetchone()
        else:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM telemetry_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return int(count)


# ---------------------------------------------------------------------------
# Row <-> event marshalling (shared with the postgres backend layout).
# ---------------------------------------------------------------------------


def _event_to_row(event: TelemetryEvent) -> tuple[Any, ...]:
    return (
        str(event.event_uuid),
        event.ts_utc.isoformat(),
        event.run_id,
        event.module,
        event.event_type.value,
        event.schema_version,
        json.dumps(event.payload, sort_keys=True, default=str),
        str(event.parent_event_uuid) if event.parent_event_uuid else None,
        event.git_sha,
    )


def _row_to_event(row: tuple[Any, ...]) -> TelemetryEvent:
    (
        _id,
        event_uuid,
        ts_utc,
        run_id,
        module,
        event_type,
        schema_version,
        payload_json,
        parent_event_uuid,
        git_sha,
    ) = row
    return TelemetryEvent(
        event_uuid=UUID(event_uuid),
        ts_utc=datetime.fromisoformat(ts_utc),
        run_id=run_id,
        module=module,
        event_type=EventType(event_type),
        schema_version=int(schema_version),
        payload=json.loads(payload_json) if payload_json else {},
        parent_event_uuid=UUID(parent_event_uuid) if parent_event_uuid else None,
        git_sha=git_sha,
    )


def _build_select(
    *,
    run_id: str | None,
    event_types: list[EventType] | None,
    since_utc: str | None,
    until_utc: str | None,
    limit: int | None,
    placeholder: str,
) -> tuple[str, list[Any]]:
    """Shared SQL builder so the SQLite and Postgres backends produce
    identical projection / filter semantics."""

    where: list[str] = []
    params: list[Any] = []
    if run_id is not None:
        where.append(f"run_id = {placeholder}")
        params.append(run_id)
    if event_types:
        marks = ",".join(placeholder for _ in event_types)
        where.append(f"event_type IN ({marks})")
        params.extend(t.value for t in event_types)
    if since_utc is not None:
        where.append(f"ts_utc >= {placeholder}")
        params.append(since_utc)
    if until_utc is not None:
        where.append(f"ts_utc <= {placeholder}")
        params.append(until_utc)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    tail = f" LIMIT {int(limit)}" if limit else ""
    sql = (
        "SELECT id, event_uuid, ts_utc, run_id, module, event_type, "
        "schema_version, payload_json, parent_event_uuid, git_sha "
        f"FROM telemetry_events{clause} ORDER BY ts_utc ASC, id ASC{tail}"
    )
    return sql, params


__all__ = ["SqliteBackend"]
