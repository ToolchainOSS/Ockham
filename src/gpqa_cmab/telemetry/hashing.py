"""Hashing helpers for telemetry artifacts and payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any | None) -> str | None:
    if value is None:
        return None
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256_text(payload)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)
