"""JSONL read/write and artifact metadata records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from gpqa_cmab.telemetry.hashing import _line_count, file_sha256
from gpqa_cmab.telemetry.recorder import _emit_artifact_event


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, BaseModel):
                handle.write(row.model_dump_json() + "\n")
            else:
                handle.write(json.dumps(row) + "\n")
            row_count += 1
    _emit_artifact_event(path, kind="jsonl", rows=row_count)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def artifact_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
    }
    if path.suffix == ".jsonl":
        record["jsonl_rows"] = _line_count(path)
    return record
