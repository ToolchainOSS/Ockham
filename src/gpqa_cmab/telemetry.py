from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from gpqa_cmab.schemas import (
    AgentId,
    AgentRole,
    AggregateTelemetry,
    CallTelemetry,
    LLMRequest,
    LLMResponse,
)

if TYPE_CHECKING:
    from gpqa_cmab.telemetry_db import TelemetryRecorder

_RECORDED_ENV_VARS = (
    "LLM_PROVIDER",
    "MAIN_MODEL",
    "SUBAGENT_MODEL",
    "SELF_CONSISTENCY_MODEL",
    "REASONING_EFFORT",
    "LLM_USE_RESPONSES_API",
    "MAX_OUTPUT_TOKENS",
    "OPENAI_API_KEY",
    "OPENAI_API_KEYS",
    "LLM_API_KEY",
    "OPENAI_BASE_URL",
    "LLM_BASE_URL",
    "OPENAI_ORGANIZATION",
    "OPENAI_KEY_COOLDOWN_S",
    "OPENAI_MAX_RETRIES",
    "OPENAI_MAX_WAIT_S",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "LLM_DEFAULT_HEADERS",
    "LLM_TIMEOUT_S",
    "LAMBDA_TOKEN",
    "LAMBDA_CALL",
    "COST_INPUT_USD_PER_1M_TOKENS",
    "COST_CACHED_INPUT_USD_PER_1M_TOKENS",
    "COST_OUTPUT_USD_PER_1M_TOKENS",
    "MAX_TOTAL_API_CALLS",
    "MAX_TOTAL_COST_USD",
    "LLM_JSON_MAX_RETRIES",
)

_SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "HEADER")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TelemetryLogger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.records: list[CallTelemetry] = []
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, telemetry: CallTelemetry) -> CallTelemetry:
        self.records.append(telemetry)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(telemetry.model_dump_json() + "\n")
        _emit_llm_event(telemetry)
        return telemetry

    def record(
        self,
        *,
        response: LLMResponse,
        experiment_id: str,
        question_id: str,
        agent_type: AgentRole,
        subset_id: str,
        model: str,
        prompt_version: str,
        temperature: float,
        request: LLMRequest | None = None,
        attempt: int = 1,
        schema_name: str | None = None,
        success: bool = True,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> CallTelemetry:
        telemetry = CallTelemetry(
            experiment_id=experiment_id,
            question_id=question_id,
            agent_type=agent_type,
            subset_id=subset_id,
            model=model,
            prompt_version=prompt_version,
            temperature=temperature,
            attempt=attempt,
            prompt_text=redact_known_secrets(request.prompt) if request else None,
            prompt_sha256=_sha256_text(request.prompt) if request else None,
            prompt_chars=len(request.prompt) if request else None,
            response_text=redact_known_secrets(response.content),
            response_sha256=_sha256_text(response.content),
            response_chars=len(response.content),
            raw_response=redact_known_secrets_in_value(response.raw_response),
            raw_response_sha256=_sha256_json(response.raw_response),
            schema_name=schema_name,
            request_metadata=_redact_metadata(request.metadata) if request else {},
            request_metadata_keys=sorted(request.metadata) if request else [],
            usage=response.usage,
            latency_ms=response.latency_ms,
            success=success,
            error_type=error_type,
            error_message=redact_known_secrets(error_message),
        )
        return self.append(telemetry)

    def records_since(self, start_index: int) -> list[CallTelemetry]:
        return self.records[start_index:]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any | None) -> str | None:
    if value is None:
        return None
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256_text(payload)


def aggregate_usage(
    records: Iterable[CallTelemetry],
    *,
    experiment_id: str,
    question_id: str,
    subset_id: str,
    selected_subagents: list[AgentId],
) -> AggregateTelemetry:
    rows = list(records)
    subagent_tokens: dict[str, int] = dict.fromkeys("ABCD", 0)
    main_tokens = 0
    for row in rows:
        if row.agent_type == "main":
            main_tokens += row.usage.total_tokens
        elif row.agent_type in subagent_tokens:
            subagent_tokens[row.agent_type] += row.usage.total_tokens
    return AggregateTelemetry(
        experiment_id=experiment_id,
        question_id=question_id,
        subset_id=subset_id,
        selected_subagents=selected_subagents,
        total_prompt_tokens=sum(row.usage.prompt_tokens for row in rows),
        total_completion_tokens=sum(row.usage.completion_tokens for row in rows),
        total_tokens=sum(row.usage.total_tokens for row in rows),
        main_tokens=main_tokens,
        subagent_tokens=subagent_tokens,
        num_subagent_calls=len(selected_subagents),
        latency_total_ms=sum(row.latency_ms for row in rows),
        estimated_cost_usd=0.0,
    )


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def summarize_traces(paths: Iterable[Path]) -> dict[str, object]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            rows.extend(read_jsonl(path))
    return summarize_trace_rows(rows)


def summarize_trace_rows(
    rows: Iterable[dict[str, Any] | CallTelemetry],
) -> dict[str, object]:
    normalized = [
        row.model_dump(mode="json") if hasattr(row, "model_dump") else row
        for row in rows
    ]
    by_agent: dict[str, int] = {}
    by_model: dict[str, int] = {}
    error_types: dict[str, int] = {}
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    reasoning_tokens = 0
    cached_prompt_tokens = 0
    prompt_audio_tokens = 0
    completion_audio_tokens = 0
    estimated_usage_rows = 0
    successes = 0
    failures = 0
    max_attempt = 0
    for row in normalized:
        agent = str(row.get("agent_type", "unknown"))
        model = str(row.get("model", "unknown"))
        by_agent[agent] = by_agent.get(agent, 0) + 1
        by_model[model] = by_model.get(model, 0) + 1
        usage = row.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
        reasoning_tokens += int(usage.get("reasoning_tokens") or 0)
        cached_prompt_tokens += int(usage.get("cached_prompt_tokens") or 0)
        prompt_audio_tokens += int(usage.get("prompt_audio_tokens") or 0)
        completion_audio_tokens += int(usage.get("completion_audio_tokens") or 0)
        if usage.get("estimated"):
            estimated_usage_rows += 1
        if row.get("success"):
            successes += 1
        else:
            failures += 1
            error_type = str(row.get("error_type") or "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        max_attempt = max(max_attempt, int(row.get("attempt") or 0))
    return {
        "call_rows": len(normalized),
        "successes": successes,
        "failures": failures,
        "max_attempt": max_attempt,
        "by_agent": dict(sorted(by_agent.items())),
        "by_model": dict(sorted(by_model.items())),
        "error_types": dict(sorted(error_types.items())),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "prompt_audio_tokens": prompt_audio_tokens,
            "completion_audio_tokens": completion_audio_tokens,
            "estimated_rows": estimated_usage_rows,
        },
    }


def sanitized_environment() -> dict[str, object]:
    return {
        name: _sanitize_env_value(name, os.environ.get(name))
        for name in _RECORDED_ENV_VARS
    }


def redact_known_secrets(text: str | None) -> str | None:
    if text is None:
        return None
    redacted = text
    for name in _RECORDED_ENV_VARS:
        value = os.environ.get(name)
        if not value or not _is_secret_env(name):
            continue
        for secret in _split_secret_values(value):
            if len(secret) >= 8:
                redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def redact_known_secrets_in_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_known_secrets(value)
    if isinstance(value, list):
        return [redact_known_secrets_in_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_known_secrets_in_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): redact_known_secrets_in_value(item) for key, item in value.items()
        }
    return value


def _redact_metadata(metadata: dict[str, str]) -> dict[str, str]:
    return {key: redact_known_secrets(value) or "" for key, value in metadata.items()}


def _sanitize_env_value(name: str, value: str | None) -> dict[str, object]:
    if value is None:
        return {"set": False}
    if _is_secret_env(name):
        parts = _split_secret_values(value)
        return {
            "set": True,
            "redacted": True,
            "items": len(parts),
            "sha256": _sha256_text(value),
            "item_sha256": [_sha256_text(part) for part in parts],
            "length": len(value),
        }
    return {"set": True, "value": value}


def _is_secret_env(name: str) -> bool:
    return any(marker in name for marker in _SECRET_ENV_MARKERS)


def _split_secret_values(value: str) -> list[str]:
    parts: list[str] = []
    for chunk in value.replace("\n", ",").split(","):
        parts.extend(part for part in chunk.split() if part)
    return parts or [value]


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


# ---------------------------------------------------------------------------
# Optional dual-write into the durable telemetry DB.
#
# These helpers are a *side channel*: they only fire when a recorder has
# been explicitly installed via ``telemetry_db.set_recorder(...)``. Tests and
# library callers that never install a recorder get the legacy JSONL-only
# behaviour and pay no DB cost. The CLI's ``main()`` installs the recorder
# so every command writes durably to SQLite (or Postgres) without any of the
# individual call sites needing to know.
# ---------------------------------------------------------------------------


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
