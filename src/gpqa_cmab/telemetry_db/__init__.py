"""Durable database-backed telemetry layer for ``gpqa_cmab``.

Goal: every research-relevant action/event from every module is persisted to
an append-only event store (SQLite by default, Postgres optional) with
microsecond UTC timestamps and full payloads. The store is the *source of
truth*; the JSONL files under ``artifacts/`` are a grep-friendly cache.

If any coding agent or process corrupts/overwrites the JSONL files, the
canonical research data is still safely in the event store and the
artefacts can be reconstructed via
:mod:`gpqa_cmab.telemetry_db.reconstruct`.

Quick start::

    from gpqa_cmab.telemetry_db import EventType, get_recorder

    rec = get_recorder()
    with rec.run(command="my-experiment"):
        rec.record(EventType.LOG_RECORD, {"hello": "world"})
"""

from __future__ import annotations

from gpqa_cmab.telemetry_db.backend import BackendKind, TelemetryBackend
from gpqa_cmab.telemetry_db.config import (
    DatabaseConfig,
    open_backend,
    parse_database_url,
    resolve_database_url,
)
from gpqa_cmab.telemetry_db.reconstruct import (
    ReconstructionReport,
    RunSummary,
    list_runs,
    reconstruct_bandit_replay,
    reconstruct_factorial,
    reconstruct_run_summary,
    verify_integrity,
)
from gpqa_cmab.telemetry_db.recorder import (
    TelemetryRecorder,
    get_active_recorder,
    get_recorder,
    set_recorder,
)
from gpqa_cmab.telemetry_db.schema import (
    SCHEMA_VERSION,
    EventType,
    TelemetryEvent,
    current_schema_version,
    ddl_statements,
)

__all__ = [
    "SCHEMA_VERSION",
    "BackendKind",
    "DatabaseConfig",
    "EventType",
    "ReconstructionReport",
    "RunSummary",
    "TelemetryBackend",
    "TelemetryEvent",
    "TelemetryRecorder",
    "current_schema_version",
    "ddl_statements",
    "get_active_recorder",
    "get_recorder",
    "list_runs",
    "open_backend",
    "parse_database_url",
    "reconstruct_bandit_replay",
    "reconstruct_factorial",
    "reconstruct_run_summary",
    "resolve_database_url",
    "set_recorder",
    "verify_integrity",
]
