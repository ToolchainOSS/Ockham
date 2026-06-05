"""Pydantic schemas + SQL DDL for the durable telemetry event store.

Design: append-only event sourcing. Every action/event from every module is
recorded as an immutable row in a single ``telemetry_events`` table. The full
research history can be reconstructed from this table alone — the JSONL files
under ``artifacts/`` become a convenience cache rather than the source of
truth.

The ``EventType`` enum is closed (no string fall-through). Adding a new event
type requires a code change *and* a schema-version bump in
:func:`current_schema_version`, which is recorded on every event row. This
makes the event log forward-debuggable: a future reader always knows which
payload shape to expect.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

SCHEMA_VERSION: Final[int] = 1
"""Bump on every backwards-incompatible payload shape change. Recorded on
every row so historical events stay decodable."""


def current_schema_version() -> int:
    return SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Closed sum of event kinds — make-invalid-states-unrepresentable
# ---------------------------------------------------------------------------


class EventType(StrEnum):
    """Closed enumeration of every event the telemetry layer can record.

    Naming convention: ``<module>.<verb>`` in past tense for completed
    actions, present tense for state-change requests. New types require a
    code change so the type checker can find every producer / consumer.
    """

    # --- run lifecycle --------------------------------------------------
    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    RUN_FAILED = "run.failed"

    # --- LLM boundary ---------------------------------------------------
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_RETRY = "llm.retry"
    LLM_ERROR = "llm.error"
    LLM_JSON_RETRY = "llm.json_retry"

    # --- agents ---------------------------------------------------------
    SUBAGENT_INVOKED = "agent.subagent_invoked"
    MAIN_INTEGRATOR_INVOKED = "agent.main_integrator_invoked"
    SELF_CONSISTENCY_INVOKED = "agent.self_consistency_invoked"

    # --- factorial experiment ------------------------------------------
    FACTORIAL_ROW = "factorial.row"
    FACTORIAL_AGGREGATE = "factorial.aggregate"

    # --- bandit replay --------------------------------------------------
    BANDIT_STEP = "bandit.step"
    BANDIT_SUMMARY = "bandit.summary"
    BANDIT_BENCHMARK = "bandit.benchmark"

    # --- GFN ------------------------------------------------------------
    GFN_TRAIN_ITER = "gfn.train_iter"
    GFN_TRAIN_DONE = "gfn.train_done"
    GFN_EVAL = "gfn.evaluation"

    # --- cost / budget --------------------------------------------------
    BUDGET_TICK = "cost.budget_tick"
    BUDGET_EXCEEDED = "cost.budget_exceeded"

    # --- artifacts ------------------------------------------------------
    ARTIFACT_WRITTEN = "artifact.written"
    MANIFEST_WRITTEN = "artifact.manifest_written"

    # --- diagnostics ----------------------------------------------------
    LOG_RECORD = "diag.log"
    """Free-form structured log line for diagnostics. Use sparingly — prefer
    a dedicated event type."""


# ---------------------------------------------------------------------------
# The single event row
# ---------------------------------------------------------------------------


class TelemetryEvent(BaseModel):
    """A single immutable, append-only event row.

    Frozen Pydantic model — the database is the only place an event can
    "change", and only by appending a *new* event. No row is ever mutated.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_uuid: UUID = Field(default_factory=uuid4)
    """Idempotency key. Re-inserting the same UUID is a no-op (UNIQUE)."""

    ts_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    """Server-local capture timestamp, microsecond precision, always UTC."""

    run_id: str
    """Groups events that belong to a single CLI invocation / experiment."""

    module: str
    """Dotted module identifier of the producer, e.g. ``llm.client``."""

    event_type: EventType
    """One of the closed :class:`EventType` values."""

    schema_version: int = Field(default_factory=current_schema_version)
    """Payload shape version. Bumped on breaking payload changes."""

    payload: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary structured data. SHOULD be JSON-serialisable scalars only."""

    parent_event_uuid: UUID | None = None
    """Optional causal link, e.g. ``LLM_RESPONSE.parent = LLM_REQUEST``."""

    git_sha: str | None = None
    """Optional git HEAD at capture time for reproducibility."""


# ---------------------------------------------------------------------------
# DDL — kept dialect-conditional so SQLite and Postgres share one source.
# ---------------------------------------------------------------------------


def ddl_statements(dialect: str) -> list[str]:
    """Return the CREATE TABLE / INDEX statements for ``dialect``.

    Dialects: ``"sqlite"`` or ``"postgres"``.
    """

    if dialect == "sqlite":
        return _SQLITE_DDL
    if dialect == "postgres":
        return _POSTGRES_DDL
    raise ValueError(f"unknown dialect: {dialect!r}")


def _index_ddl(column: str) -> str:
    return (
        f"CREATE INDEX IF NOT EXISTS ix_telemetry_events_{column} "
        f"ON telemetry_events ({column})"
    )


_INDEX_COLUMNS: list[str] = ["ts_utc", "run_id", "event_type", "module"]


_SQLITE_DDL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS telemetry_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        event_uuid      TEXT    NOT NULL UNIQUE,
        ts_utc          TEXT    NOT NULL,
        run_id          TEXT    NOT NULL,
        module          TEXT    NOT NULL,
        event_type      TEXT    NOT NULL,
        schema_version  INTEGER NOT NULL,
        payload_json    TEXT    NOT NULL,
        parent_event_uuid TEXT,
        git_sha         TEXT
    )
    """,
    *(_index_ddl(col) for col in _INDEX_COLUMNS),
    """
    CREATE TABLE IF NOT EXISTS telemetry_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
]


_POSTGRES_DDL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS telemetry_events (
        id                BIGSERIAL    PRIMARY KEY,
        event_uuid        UUID         NOT NULL UNIQUE,
        ts_utc            TIMESTAMPTZ  NOT NULL,
        run_id            TEXT         NOT NULL,
        module            TEXT         NOT NULL,
        event_type        TEXT         NOT NULL,
        schema_version    INTEGER      NOT NULL,
        payload_json      JSONB        NOT NULL,
        parent_event_uuid UUID,
        git_sha           TEXT
    )
    """,
    *(_index_ddl(col) for col in _INDEX_COLUMNS),
    """
    CREATE TABLE IF NOT EXISTS telemetry_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
]


__all__ = [
    "SCHEMA_VERSION",
    "EventType",
    "TelemetryEvent",
    "current_schema_version",
    "ddl_statements",
]
