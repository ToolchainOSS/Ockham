"""Optional dual-write into the durable telemetry DB.

These helpers are a *side channel*: they only fire when a recorder has been
explicitly installed via ``telemetry_db.set_recorder(...)``. Tests and library
callers that never install a recorder get the legacy JSONL-only behaviour and
pay no DB cost. The CLI's ``main()`` installs the recorder so every command
writes durably to SQLite (or Postgres) without any of the individual call
sites needing to know.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gpqa_cmab.schemas import CallTelemetry

if TYPE_CHECKING:
    from gpqa_cmab.telemetry_db import TelemetryRecorder


def _emit_llm_event(row: CallTelemetry) -> None:
    rec = _maybe_recorder()
    if rec is None or rec.run_id is None:
        return
    from gpqa_cmab.telemetry_db import EventType  # local import: avoid cycles

    try:  # noqa: SIM105 — broad except is intentional: telemetry must NEVER raise
        rec.record(
            EventType.LLM_RESPONSE,
            row.model_dump(mode="json"),
            module="gpqa_cmab.llm",
        )
    except Exception:
        pass


def _emit_artifact_event(path: Path, *, kind: str, rows: int | None = None) -> None:
    rec = _maybe_recorder()
    if rec is None or rec.run_id is None:
        return
    from gpqa_cmab.telemetry_db import EventType

    try:  # noqa: SIM105
        rec.record(
            EventType.ARTIFACT_WRITTEN,
            {
                "path": str(path),
                "bytes": path.stat().st_size if path.exists() else None,
                "kind": kind,
                "rows": rows,
            },
            module="gpqa_cmab.telemetry",
        )
    except Exception:
        pass


def _emit_manifest_event(path: Path, *, command: str, status: str) -> None:
    rec = _maybe_recorder()
    if rec is None or rec.run_id is None:
        return
    from gpqa_cmab.telemetry_db import EventType

    try:  # noqa: SIM105
        rec.record(
            EventType.MANIFEST_WRITTEN,
            {"path": str(path), "command": command, "status": status},
            module="gpqa_cmab.telemetry",
        )
    except Exception:
        pass


def _maybe_recorder() -> TelemetryRecorder | None:
    try:
        from gpqa_cmab.telemetry_db import get_active_recorder
    except Exception:
        return None
    try:
        return get_active_recorder()
    except Exception:
        return None
