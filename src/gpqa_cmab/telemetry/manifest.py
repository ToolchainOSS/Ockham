"""Run-manifest writing plus prompt/source inventory records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from gpqa_cmab.telemetry.io import artifact_record
from gpqa_cmab.telemetry.recorder import _emit_manifest_event
from gpqa_cmab.telemetry.redaction import sanitized_environment
from gpqa_cmab.telemetry.summary import summarize_traces

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def prompt_inventory(prompt_dir: Path | None = None) -> list[dict[str, object]]:
    if prompt_dir is None:
        from gpqa_cmab.prompts import PROMPTS_DIR

        prompt_dir = PROMPTS_DIR
    if not prompt_dir.exists():
        return []
    return [artifact_record(path) for path in sorted(prompt_dir.glob("*.txt"))]


def source_inventory(project_root: Path = _PROJECT_ROOT) -> list[dict[str, object]]:
    paths: list[Path] = []
    src_root = project_root / "src" / "gpqa_cmab"
    if src_root.exists():
        paths.extend(sorted(src_root.rglob("*.py")))
    for name in ("pyproject.toml", "uv.lock"):
        path = project_root / name
        if path.exists():
            paths.append(path)
    return [artifact_record(path) for path in paths]


def write_run_manifest(
    path: Path,
    *,
    command: str,
    started_utc: str,
    status: str,
    argv: list[str] | None = None,
    inputs: Iterable[Path] = (),
    artifacts: Iterable[Path] = (),
    traces: Iterable[Path] = (),
    settings: dict[str, object] | None = None,
    budget: Mapping[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    existing_artifacts = [item for item in artifacts if item.exists()]
    existing_traces = [item for item in traces if item.exists()]
    payload: dict[str, object] = {
        "schema_version": 2,
        "command": command,
        "argv": argv or [],
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "inputs": [artifact_record(item) for item in inputs if item.exists()],
        "prompts": prompt_inventory(),
        "source": source_inventory(),
        "artifacts": [artifact_record(item) for item in existing_artifacts],
        "traces": [artifact_record(item) for item in existing_traces],
        "trace_summary": summarize_traces(existing_traces),
        "environment": sanitized_environment(),
        "settings": settings or {},
        "budget": budget or {},
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _emit_manifest_event(path, command=command, status=status)
