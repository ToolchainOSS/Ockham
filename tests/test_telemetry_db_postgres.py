"""Integration tests for the Postgres telemetry backend.

These tests are SKIPPED unless ``GPQA_TELEMETRY_PG_DSN`` is set, so the
default test run stays network-free per AGENTS.md.

To run locally::

    GPQA_TELEMETRY_PG_DSN='postgresql://postgres:postgres@localhost:5432/gpqa_test' \\
        uv run pytest tests/test_telemetry_db_postgres.py -v
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

DSN = os.environ.get("GPQA_TELEMETRY_PG_DSN")
pytestmark = pytest.mark.skipif(
    not DSN,
    reason="Set GPQA_TELEMETRY_PG_DSN to run Postgres integration tests.",
)

psycopg = pytest.importorskip("psycopg")

from gpqa_cmab.telemetry_db import EventType, TelemetryEvent  # noqa: E402
from gpqa_cmab.telemetry_db.postgres_backend import PostgresBackend  # noqa: E402


@pytest.fixture()
def backend() -> PostgresBackend:
    schema = f"gpqa_test_{uuid4().hex[:8]}"
    with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
    dsn = f"{DSN}?options=-csearch_path%3D{schema}"
    b = PostgresBackend(dsn)
    b.initialize()
    yield b
    b.close()
    with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")


def test_pg_append_round_trip(backend: PostgresBackend) -> None:
    ev = TelemetryEvent(
        run_id="r1", module="t", event_type=EventType.LOG_RECORD, payload={"x": 1}
    )
    backend.append(ev)
    out = list(backend.iter_events())
    assert len(out) == 1
    assert out[0].payload == {"x": 1}


def test_pg_idempotent_uuid(backend: PostgresBackend) -> None:
    ev = TelemetryEvent(
        run_id="r", module="t", event_type=EventType.LOG_RECORD, payload={}
    )
    backend.append(ev)
    backend.append(ev)
    assert backend.count_events() == 1
