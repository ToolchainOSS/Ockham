from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from gpqa_cmab.schemas import CallTelemetry, LLMRequest, LLMResponse, Usage

if TYPE_CHECKING:
    from gpqa_cmab.llm.base import LLMClient
    from gpqa_cmab.telemetry import TelemetryLogger

ModelT = TypeVar("ModelT", bound=BaseModel)

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        return _JSON_FENCE_RE.sub("", stripped).strip()
    return stripped


def _schema_hint(model_type: type[BaseModel]) -> str:
    """Render a strict JSON schema hint for inclusion in an LLM prompt.

    Embedding the actual Pydantic JSON schema (instead of a prose description
    of the expected fields) dramatically reduces field-name hallucination by
    real LLMs while leaving mock clients unaffected.
    """
    schema = model_type.model_json_schema()
    return (
        "\n\nReturn ONLY a single JSON object that strictly conforms to the "
        "following JSON Schema. Do not include any keys not listed in the "
        "schema. Do not wrap the JSON in markdown fences or commentary.\n"
        "```json\n"
        f"{json.dumps(schema, sort_keys=True)}\n"
        "```"
    )


def parse_json_with_retries(
    invoke: Callable[[str], str],
    request: LLMRequest,
    model_type: type[ModelT],
    max_retries: int = 2,
) -> ModelT:
    prompt = request.prompt + _schema_hint(model_type)
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        content = invoke(prompt)
        try:
            return model_type.model_validate(json.loads(_strip_code_fences(content)))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            prompt = (
                f"{request.prompt}{_schema_hint(model_type)}\n\n"
                f"Previous JSON was invalid on attempt {attempt + 1}: "
                f"{exc}. Return only valid JSON."
            )
    raise ValueError(f"Failed to parse JSON after retries: {last_error}")


def _zero_response() -> LLMResponse:
    return LLMResponse(
        content="",
        usage=Usage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0, estimated=True
        ),
        latency_ms=0,
    )


def complete_validated(
    client: LLMClient,
    request: LLMRequest,
    model_type: type[ModelT],
    *,
    telemetry: TelemetryLogger,
    record_kwargs: dict[str, Any],
    max_retries: int = 2,
) -> tuple[ModelT, CallTelemetry]:
    """Execute an LLM call with retries, JSON validation, and telemetry.

    Logs each API call before and after execution. Records `success=False`
    telemetry rows for API failures and for malformed JSON attempts. Raises
    after `max_retries` consecutive parse failures.
    """
    prompt = request.prompt + _schema_hint(model_type)
    last_error: Exception | None = None
    last_row: CallTelemetry | None = None
    for attempt in range(max_retries + 1):
        attempt_request = request.model_copy(update={"prompt": prompt})
        logger.info(
            "llm_request_start agent=%s question=%s model=%s attempt=%d",
            record_kwargs.get("agent_type"),
            record_kwargs.get("question_id"),
            record_kwargs.get("model"),
            attempt + 1,
        )
        try:
            response = client.complete(attempt_request)
        except Exception as exc:  # noqa: BLE001 - boundary needs broad catch
            last_row = telemetry.record(
                response=_zero_response(),
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
                **record_kwargs,
            )
            logger.exception(
                "llm_request_failed agent=%s question=%s attempt=%d",
                record_kwargs.get("agent_type"),
                record_kwargs.get("question_id"),
                attempt + 1,
            )
            raise
        logger.info(
            "llm_request_complete agent=%s question=%s tokens=%d latency_ms=%d",
            record_kwargs.get("agent_type"),
            record_kwargs.get("question_id"),
            response.usage.total_tokens,
            response.latency_ms,
        )
        try:
            parsed = model_type.model_validate(
                json.loads(_strip_code_fences(response.content))
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            last_row = telemetry.record(
                response=response,
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
                **record_kwargs,
            )
            prompt = (
                f"{request.prompt}{_schema_hint(model_type)}\n\n"
                f"Previous JSON was invalid on attempt {attempt + 1}: "
                f"{exc}. Return only valid JSON conforming to the schema."
            )
            continue
        row = telemetry.record(response=response, success=True, **record_kwargs)
        return parsed, row
    assert last_row is not None  # noqa: S101
    raise ValueError(
        f"Failed to parse JSON after {max_retries} retries for "
        f"{record_kwargs.get('agent_type')}: {last_error}"
    )
