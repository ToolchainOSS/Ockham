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

# Valid characters that may follow a backslash inside a JSON string. Any other
# backslash (e.g. LaTeX `\frac`, `\sqrt`, `\alpha`) is technically invalid and
# must be escaped to `\\` before json.loads will accept it.
_VALID_JSON_ESCAPES = set('"\\/bfnrtu')

# Matches trailing commas just before a closing bracket/brace, which some LLMs
# emit and which strict JSON forbids.
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_code_fences(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        return _JSON_FENCE_RE.sub("", stripped).strip()
    return stripped


def _extract_first_json_object(text: str) -> str:
    """Return the first balanced `{...}` substring, ignoring braces inside
    strings. If no balanced object is found, return the original text.
    """
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text


def _fix_invalid_backslash_escapes(text: str) -> str:
    """Inside JSON string literals, escape any backslash not already part of a
    valid JSON escape sequence. Outside strings, leave content unchanged.
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue
        # Inside a string.
        if ch == '"':
            out.append(ch)
            in_string = False
            i += 1
            continue
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        # ch == '\\': look at next char.
        nxt = text[i + 1] if i + 1 < n else ""
        if nxt in _VALID_JSON_ESCAPES:
            out.append(ch)
            out.append(nxt)
            i += 2
        else:
            # Escape the lone backslash so json.loads accepts it.
            out.append("\\\\")
            i += 1
    return "".join(out)


def _robust_json_loads(content: str) -> Any:
    """Parse JSON from an LLM response, tolerating common formatting issues:

    * markdown code fences
    * leading/trailing prose around the JSON object
    * trailing commas before `}` or `]`
    * unescaped backslashes (e.g. LaTeX) inside string values
    """
    candidate = _strip_code_fences(content)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    extracted = _extract_first_json_object(candidate)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass
    repaired = _TRAILING_COMMA_RE.sub(r"\1", extracted)
    repaired = _fix_invalid_backslash_escapes(repaired)
    return json.loads(repaired)


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


def _retry_prompt(
    base_prompt: str,
    model_type: type[BaseModel],
    attempt: int,
    exc: Exception,
) -> str:
    return (
        f"{base_prompt}{_schema_hint(model_type)}\n\n"
        f"Previous response on attempt {attempt} could not be parsed: "
        f"{exc}. Common mistakes to avoid:\n"
        "- Do NOT wrap the JSON in markdown code fences.\n"
        "- Do NOT add commentary before or after the JSON object.\n"
        "- Inside string values, escape every backslash as \\\\ "
        "(e.g. write LaTeX `\\\\frac` not `\\frac`).\n"
        "- Do NOT include trailing commas before } or ].\n"
        "- Use only the field names defined in the schema.\n"
        "Return ONLY a single valid JSON object conforming to the schema."
    )


def parse_json_with_retries(
    invoke: Callable[[str], str],
    request: LLMRequest,
    model_type: type[ModelT],
    max_retries: int = 3,
) -> ModelT:
    prompt = request.prompt + _schema_hint(model_type)
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        content = invoke(prompt)
        try:
            return model_type.model_validate(_robust_json_loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            prompt = _retry_prompt(request.prompt, model_type, attempt + 1, exc)
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
    max_retries: int = 3,
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
            parsed = model_type.model_validate(_robust_json_loads(response.content))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            last_row = telemetry.record(
                response=response,
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
                **record_kwargs,
            )
            prompt = _retry_prompt(request.prompt, model_type, attempt + 1, exc)
            continue
        row = telemetry.record(response=response, success=True, **record_kwargs)
        return parsed, row
    assert last_row is not None  # noqa: S101
    raise ValueError(
        f"Failed to parse JSON after {max_retries} retries for "
        f"{record_kwargs.get('agent_type')}: {last_error}"
    )
