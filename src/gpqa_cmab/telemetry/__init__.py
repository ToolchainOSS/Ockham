"""Telemetry: structured logging, redaction, manifests, and trace summaries.

This package preserves the original flat ``gpqa_cmab.telemetry`` import
surface. All previously top-level names remain importable from here.
"""

from __future__ import annotations

from gpqa_cmab.telemetry.hashing import (
    _sha256_json,
    _sha256_text,
    file_sha256,
)
from gpqa_cmab.telemetry.io import (
    artifact_record,
    read_jsonl,
    write_jsonl,
)
from gpqa_cmab.telemetry.logger import TelemetryLogger
from gpqa_cmab.telemetry.manifest import (
    prompt_inventory,
    source_inventory,
    write_run_manifest,
)
from gpqa_cmab.telemetry.redaction import (
    redact_known_secrets,
    redact_known_secrets_in_value,
    sanitized_environment,
)
from gpqa_cmab.telemetry.summary import (
    aggregate_usage,
    summarize_trace_rows,
    summarize_traces,
)

__all__ = [
    "TelemetryLogger",
    "_sha256_json",
    "_sha256_text",
    "aggregate_usage",
    "artifact_record",
    "file_sha256",
    "prompt_inventory",
    "read_jsonl",
    "redact_known_secrets",
    "redact_known_secrets_in_value",
    "sanitized_environment",
    "source_inventory",
    "summarize_trace_rows",
    "summarize_traces",
    "write_jsonl",
    "write_run_manifest",
]
