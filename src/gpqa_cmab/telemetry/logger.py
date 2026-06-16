"""The ``TelemetryLogger`` that records LLM calls to JSONL (and optionally DB)."""

from __future__ import annotations

from pathlib import Path

from gpqa_cmab.schemas import (
    AgentRole,
    CallTelemetry,
    LLMRequest,
    LLMResponse,
)
from gpqa_cmab.telemetry.hashing import _sha256_json, _sha256_text
from gpqa_cmab.telemetry.recorder import _emit_llm_event
from gpqa_cmab.telemetry.redaction import (
    _redact_metadata,
    redact_known_secrets,
    redact_known_secrets_in_value,
)


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
