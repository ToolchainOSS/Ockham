"""Reconstruct research artefacts from the event log alone.

The goal: even if every file under ``artifacts/`` is wiped, we can rebuild
the canonical research outputs from the durable event log.

This module provides:

* :func:`reconstruct_factorial` — rebuild the per-question factorial JSONL.
* :func:`reconstruct_bandit_replay` — rebuild a bandit replay JSONL.
* :func:`reconstruct_run_summary` — group all events for a single ``run_id``
  into a :class:`ReconstructionReport` (Pydantic) summarising what happened.
* :func:`list_runs` — enumerate every ``run_id`` known to the DB with start
  / end timestamps and event counts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gpqa_cmab.telemetry_db.backend import TelemetryBackend
from gpqa_cmab.telemetry_db.schema import EventType, TelemetryEvent


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    started_utc: datetime | None
    finished_utc: datetime | None
    status: str  # "running" | "finished" | "failed" | "unknown"
    event_count: int
    by_event_type: dict[str, int] = Field(default_factory=dict)
    command: str | None = None


class ReconstructionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    factorial_rows: int = 0
    bandit_steps: int = 0
    llm_calls: int = 0
    written_artifacts: list[str] = Field(default_factory=list)
    output_path: Path | None = None


# ---------------------------------------------------------------------------
# Listing / per-run summary
# ---------------------------------------------------------------------------


def list_runs(backend: TelemetryBackend) -> list[RunSummary]:
    """Return one :class:`RunSummary` per distinct ``run_id`` in the DB."""

    grouped: dict[str, list[TelemetryEvent]] = defaultdict(list)
    for ev in backend.iter_events():
        grouped[ev.run_id].append(ev)
    return [_summarise(rid, events) for rid, events in sorted(grouped.items())]


def _summarise(run_id: str, events: list[TelemetryEvent]) -> RunSummary:
    by_type: dict[str, int] = defaultdict(int)
    started: datetime | None = None
    finished: datetime | None = None
    status = "unknown"
    command: str | None = None
    for ev in events:
        by_type[ev.event_type.value] += 1
        if ev.event_type is EventType.RUN_STARTED:
            started = ev.ts_utc
            command = str(ev.payload.get("command")) if ev.payload else None
        elif ev.event_type is EventType.RUN_FINISHED:
            finished = ev.ts_utc
            status = "finished"
        elif ev.event_type is EventType.RUN_FAILED:
            finished = ev.ts_utc
            status = "failed"
    if status == "unknown" and started is not None and finished is None:
        status = "running"
    return RunSummary(
        run_id=run_id,
        started_utc=started,
        finished_utc=finished,
        status=status,
        event_count=len(events),
        by_event_type=dict(by_type),
        command=command,
    )


# ---------------------------------------------------------------------------
# Artefact reconstruction
# ---------------------------------------------------------------------------


def reconstruct_factorial(
    backend: TelemetryBackend,
    output_path: Path,
    *,
    run_id: str | None = None,
) -> ReconstructionReport:
    """Rebuild a factorial-results JSONL from FACTORIAL_ROW events.

    If multiple events share the same ``(question_id, subset_id)``, the
    latest by ``ts_utc`` wins (last-write-wins for replays).
    """

    rows: dict[tuple[str, str], TelemetryEvent] = {}
    for ev in backend.iter_events(run_id=run_id, event_types=[EventType.FACTORIAL_ROW]):
        qid = str(ev.payload.get("question_id", ""))
        sid = str(ev.payload.get("subset_id", ""))
        rows[(qid, sid)] = ev  # later timestamps overwrite earlier
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for (_, _), ev in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            handle.write(json.dumps(ev.payload, default=str) + "\n")
    return ReconstructionReport(
        run_id=run_id or "*",
        factorial_rows=len(rows),
        output_path=output_path,
    )


def reconstruct_bandit_replay(
    backend: TelemetryBackend,
    output_path: Path,
    *,
    run_id: str,
) -> ReconstructionReport:
    """Rebuild a bandit replay JSONL from BANDIT_STEP events for one run."""

    steps = list(
        backend.iter_events(run_id=run_id, event_types=[EventType.BANDIT_STEP])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for ev in steps:
            handle.write(json.dumps(ev.payload, default=str) + "\n")
    return ReconstructionReport(
        run_id=run_id,
        bandit_steps=len(steps),
        output_path=output_path,
    )


def reconstruct_run_summary(
    backend: TelemetryBackend, run_id: str
) -> ReconstructionReport:
    """Aggregate event counts for one run."""

    llm = 0
    fact = 0
    bandit = 0
    artifacts: list[str] = []
    for ev in backend.iter_events(run_id=run_id):
        if ev.event_type is EventType.LLM_RESPONSE:
            llm += 1
        elif ev.event_type is EventType.FACTORIAL_ROW:
            fact += 1
        elif ev.event_type is EventType.BANDIT_STEP:
            bandit += 1
        elif ev.event_type is EventType.ARTIFACT_WRITTEN and (
            ev.payload and ev.payload.get("path")
        ):
            artifacts.append(str(ev.payload["path"]))
    return ReconstructionReport(
        run_id=run_id,
        factorial_rows=fact,
        bandit_steps=bandit,
        llm_calls=llm,
        written_artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


def verify_integrity(backend: TelemetryBackend) -> dict[str, Any]:
    """Cheap sanity check over the whole event store."""

    counts: dict[str, int] = defaultdict(int)
    seen_uuids: set[str] = set()
    duplicate_uuids = 0
    runs: set[str] = set()
    earliest: datetime | None = None
    latest: datetime | None = None
    for ev in backend.iter_events():
        counts[ev.event_type.value] += 1
        runs.add(ev.run_id)
        uid = str(ev.event_uuid)
        if uid in seen_uuids:
            duplicate_uuids += 1
        else:
            seen_uuids.add(uid)
        if earliest is None or ev.ts_utc < earliest:
            earliest = ev.ts_utc
        if latest is None or ev.ts_utc > latest:
            latest = ev.ts_utc
    return {
        "total_events": sum(counts.values()),
        "distinct_runs": len(runs),
        "duplicate_uuids": duplicate_uuids,
        "by_event_type": dict(sorted(counts.items())),
        "earliest_utc": earliest.isoformat() if earliest else None,
        "latest_utc": latest.isoformat() if latest else None,
    }


def iter_payloads(
    events: Iterable[TelemetryEvent],
) -> Iterable[dict[str, Any]]:
    """Convenience: yield ``ev.payload`` for each event."""
    for ev in events:
        yield ev.payload


__all__ = [
    "ReconstructionReport",
    "RunSummary",
    "iter_payloads",
    "list_runs",
    "reconstruct_bandit_replay",
    "reconstruct_factorial",
    "reconstruct_run_summary",
    "verify_integrity",
]
