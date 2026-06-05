"""Unit tests for the durable telemetry DB layer (SQLite backend).

Postgres-specific tests live in a separate, env-gated module so the default
test suite stays network-free per AGENTS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpqa_cmab.telemetry_db import (
    EventType,
    TelemetryEvent,
    TelemetryRecorder,
    list_runs,
    open_backend,
    parse_database_url,
    reconstruct_factorial,
    set_recorder,
    verify_integrity,
)
from gpqa_cmab.telemetry_db.sqlite_backend import SqliteBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "telemetry.sqlite"


@pytest.fixture()
def backend(db_path: Path) -> SqliteBackend:
    b = SqliteBackend(db_path)
    b.initialize()
    yield b
    b.close()


@pytest.fixture()
def recorder(backend: SqliteBackend) -> TelemetryRecorder:
    rec = TelemetryRecorder(backend, run_id="test-run-1")
    set_recorder(None)  # ensure clean process-wide state
    return rec


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_parse_sqlite_url_relative() -> None:
    cfg = parse_database_url("sqlite:///artifacts/telemetry.sqlite")
    assert cfg.kind == "sqlite"
    assert cfg.sqlite_path == Path("artifacts/telemetry.sqlite")


def test_parse_sqlite_url_absolute() -> None:
    cfg = parse_database_url("sqlite:////tmp/test.sqlite")
    assert cfg.kind == "sqlite"
    assert cfg.sqlite_path == Path("/tmp/test.sqlite")


def test_parse_postgres_url() -> None:
    cfg = parse_database_url("postgresql://user:pw@host:5432/db")
    assert cfg.kind == "postgres"
    assert cfg.postgres_dsn == "postgresql://user:pw@host:5432/db"


def test_parse_unknown_scheme_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        parse_database_url("mysql://localhost/db")


# ---------------------------------------------------------------------------
# Backend semantics
# ---------------------------------------------------------------------------


def test_append_persists_and_round_trips(backend: SqliteBackend) -> None:
    ev = TelemetryEvent(
        run_id="r1",
        module="test",
        event_type=EventType.LOG_RECORD,
        payload={"k": "v", "n": 42},
    )
    backend.append(ev)
    out = list(backend.iter_events())
    assert len(out) == 1
    assert out[0].event_uuid == ev.event_uuid
    assert out[0].payload == {"k": "v", "n": 42}
    assert out[0].event_type is EventType.LOG_RECORD


def test_append_is_idempotent_on_uuid(backend: SqliteBackend) -> None:
    ev = TelemetryEvent(
        run_id="r1", module="t", event_type=EventType.LOG_RECORD, payload={}
    )
    backend.append(ev)
    backend.append(ev)
    assert backend.count_events() == 1


def test_iter_events_filters(backend: SqliteBackend) -> None:
    for i in range(5):
        backend.append(
            TelemetryEvent(
                run_id="A" if i < 3 else "B",
                module="m",
                event_type=EventType.LLM_RESPONSE
                if i % 2 == 0
                else EventType.LOG_RECORD,
                payload={"i": i},
            )
        )
    a_only = list(backend.iter_events(run_id="A"))
    assert len(a_only) == 3
    llm_only = list(backend.iter_events(event_types=[EventType.LLM_RESPONSE]))
    assert {e.event_type for e in llm_only} == {EventType.LLM_RESPONSE}
    limit = list(backend.iter_events(limit=2))
    assert len(limit) == 2


def test_durability_across_reopen(db_path: Path) -> None:
    b1 = SqliteBackend(db_path)
    b1.initialize()
    b1.append(
        TelemetryEvent(
            run_id="r", module="m", event_type=EventType.LOG_RECORD, payload={"x": 1}
        )
    )
    b1.close()

    b2 = SqliteBackend(db_path)
    b2.initialize()
    rows = list(b2.iter_events())
    assert len(rows) == 1
    assert rows[0].payload == {"x": 1}
    b2.close()


def test_payload_with_non_jsonable_falls_back_to_str(backend: SqliteBackend) -> None:
    # Path is not JSON-native; the backend serialises with default=str.
    ev = TelemetryEvent(
        run_id="r",
        module="m",
        event_type=EventType.ARTIFACT_WRITTEN,
        payload={"path": Path("/tmp/x")},
    )
    backend.append(ev)
    out = list(backend.iter_events())[0]
    assert out.payload["path"] == "/tmp/x"


# ---------------------------------------------------------------------------
# Recorder + run scope
# ---------------------------------------------------------------------------


def test_recorder_run_emits_started_and_finished(backend: SqliteBackend) -> None:
    rec = TelemetryRecorder(backend)
    with rec.run(command="unit-test"):
        rec.record(EventType.LOG_RECORD, {"hello": "world"})
    events = list(backend.iter_events())
    types = [e.event_type for e in events]
    assert EventType.RUN_STARTED in types
    assert EventType.LOG_RECORD in types
    assert EventType.RUN_FINISHED in types
    assert EventType.RUN_FAILED not in types


def test_recorder_run_emits_failed_on_exception(backend: SqliteBackend) -> None:
    rec = TelemetryRecorder(backend)
    with pytest.raises(ValueError), rec.run(command="unit-test"):
        raise ValueError("boom")
    failed = [e for e in backend.iter_events() if e.event_type is EventType.RUN_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["error_type"] == "ValueError"
    assert failed[0].payload["error_message"] == "boom"


def test_recorder_requires_run_id() -> None:
    backend = SqliteBackend(":memory:")
    backend.initialize()
    rec = TelemetryRecorder(backend)
    with pytest.raises(RuntimeError, match="run_id"):
        rec.record(EventType.LOG_RECORD, {})


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def test_reconstruct_factorial_round_trip(
    backend: SqliteBackend, tmp_path: Path
) -> None:
    rec = TelemetryRecorder(backend)
    with rec.run(command="reconstruct-test") as rid:
        for q in ("q1", "q2"):
            for s in ("A", "A,B", "main_only"):
                rec.record(
                    EventType.FACTORIAL_ROW,
                    {"question_id": q, "subset_id": s, "correct": True},
                )
    out = tmp_path / "rebuilt.jsonl"
    report = reconstruct_factorial(backend, out, run_id=rid)
    assert report.factorial_rows == 6
    rebuilt = [json.loads(line) for line in out.read_text().splitlines()]
    assert {(r["question_id"], r["subset_id"]) for r in rebuilt} == {
        (q, s) for q in ("q1", "q2") for s in ("A", "A,B", "main_only")
    }


def test_list_runs_groups_by_run_id(backend: SqliteBackend) -> None:
    rec = TelemetryRecorder(backend)
    with rec.run(command="first"):
        rec.record(EventType.LOG_RECORD, {"i": 1})
    with rec.run(command="second"):
        rec.record(EventType.LOG_RECORD, {"i": 2})
    summaries = list_runs(backend)
    assert len(summaries) == 2
    commands = {s.command for s in summaries}
    assert commands == {"first", "second"}
    assert all(s.status == "finished" for s in summaries)


def test_verify_integrity_reports_counts(backend: SqliteBackend) -> None:
    rec = TelemetryRecorder(backend)
    with rec.run(command="verify-test"):
        rec.record(EventType.LLM_RESPONSE, {"x": 1})
        rec.record(EventType.LLM_RESPONSE, {"x": 2})
    report = verify_integrity(backend)
    assert report["distinct_runs"] == 1
    assert report["duplicate_uuids"] == 0
    assert report["by_event_type"]["llm.response"] == 2


# ---------------------------------------------------------------------------
# Integration with the existing JSONL telemetry: dual-write
# ---------------------------------------------------------------------------


def test_telemetry_logger_dual_writes_into_db(
    backend: SqliteBackend, tmp_path: Path
) -> None:
    from gpqa_cmab.schemas import CallTelemetry, Usage
    from gpqa_cmab.telemetry import TelemetryLogger

    rec = TelemetryRecorder(backend, run_id="dual-write-run")
    set_recorder(rec)
    try:
        logger = TelemetryLogger(tmp_path / "trace.jsonl")
        logger.append(
            CallTelemetry(
                experiment_id="exp",
                question_id="q1",
                agent_type="A",
                subset_id="A",
                model="m",
                prompt_version="v1",
                temperature=0.0,
                attempt=1,
                response_text="hello",
                response_sha256="abc",
                response_chars=5,
                usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                latency_ms=10,
                success=True,
            )
        )
        db_events = [
            e
            for e in backend.iter_events(event_types=[EventType.LLM_RESPONSE])
            if e.run_id == "dual-write-run"
        ]
        assert len(db_events) == 1
        assert db_events[0].payload["question_id"] == "q1"
    finally:
        set_recorder(None)


def test_open_backend_uses_default_when_no_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Redirect to a temp file so we don't touch artifacts/telemetry.sqlite.
    db = tmp_path / "default.sqlite"
    monkeypatch.setenv("GPQA_TELEMETRY_DB_URL", f"sqlite:///{db}")
    backend = open_backend()
    assert backend.kind == "sqlite"
    backend.append(
        TelemetryEvent(
            run_id="r", module="m", event_type=EventType.LOG_RECORD, payload={}
        )
    )
    assert db.exists()
    backend.close()
