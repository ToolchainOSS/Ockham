from __future__ import annotations

import pytest

from gpqa_cmab.json_utils import complete_validated, parse_json_with_retries
from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.schemas import (
    LLMRequest,
    LLMResponse,
    MainIntegratorOutput,
    Usage,
)
from gpqa_cmab.telemetry import TelemetryLogger

VALID_JSON = (
    '{"final_answer":"A","confidence":0.8,"rationale_summary":"ok",'
    '"subagent_influence":{"A":"not_used","B":"not_used","C":"not_used",'
    '"D":"not_used"}}'
)


def _record_kwargs() -> dict:
    return {
        "experiment_id": "exp",
        "question_id": "q",
        "agent_type": "main",
        "subset_id": "main_only",
        "model": "m",
        "prompt_version": "main_integrator_v1",
        "temperature": 0.0,
    }


def test_json_parse_retry_behavior():
    calls = iter(["not json", VALID_JSON])
    parsed = parse_json_with_retries(
        lambda _: next(calls), LLMRequest(prompt="p", model="m"), MainIntegratorOutput
    )
    assert parsed.final_answer == "A"


def test_parse_strips_markdown_fences():
    fenced = f"```json\n{VALID_JSON}\n```"
    parsed = parse_json_with_retries(
        lambda _: fenced, LLMRequest(prompt="p", model="m"), MainIntegratorOutput
    )
    assert parsed.final_answer == "A"


class _ScriptedClient(LLMClient):
    def __init__(self, contents: list[str]) -> None:
        self._contents = iter(contents)

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content=next(self._contents),
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            latency_ms=1,
        )


class _RaisingClient(LLMClient):
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise RuntimeError("boom")


def test_complete_validated_retries_then_succeeds():
    client = _ScriptedClient(["not json", VALID_JSON])
    telemetry = TelemetryLogger()
    parsed, row = complete_validated(
        client,
        LLMRequest(prompt="p", model="m"),
        MainIntegratorOutput,
        telemetry=telemetry,
        record_kwargs=_record_kwargs(),
    )
    assert parsed.final_answer == "A"
    assert row.success is True
    # 2 attempts → first failed (success=False), second succeeded.
    assert [record.success for record in telemetry.records] == [False, True]
    assert telemetry.records[0].error_type == "JSONDecodeError"


def test_complete_validated_logs_api_failure_and_raises():
    telemetry = TelemetryLogger()
    with pytest.raises(RuntimeError):
        complete_validated(
            _RaisingClient(),
            LLMRequest(prompt="p", model="m"),
            MainIntegratorOutput,
            telemetry=telemetry,
            record_kwargs=_record_kwargs(),
        )
    assert len(telemetry.records) == 1
    failure = telemetry.records[0]
    assert failure.success is False
    assert failure.error_type == "RuntimeError"


def test_complete_validated_exhausts_retries_and_raises():
    client = _ScriptedClient(["x", "y", "z"])
    telemetry = TelemetryLogger()
    with pytest.raises(ValueError):
        complete_validated(
            client,
            LLMRequest(prompt="p", model="m"),
            MainIntegratorOutput,
            telemetry=telemetry,
            record_kwargs=_record_kwargs(),
            max_retries=2,
        )
    assert len(telemetry.records) == 3
    assert all(record.success is False for record in telemetry.records)


class _PromptCapturingClient(LLMClient):
    def __init__(self, content: str) -> None:
        self._content = content
        self.last_prompt: str | None = None

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.last_prompt = request.prompt
        return LLMResponse(
            content=self._content,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            latency_ms=1,
        )


def test_complete_validated_prefixes_schema_hint_for_cache_friendly_prompt():
    """Real LLMs need the schema, and providers cache long shared prefixes."""
    client = _PromptCapturingClient(VALID_JSON)
    telemetry = TelemetryLogger()
    complete_validated(
        client,
        LLMRequest(prompt="ORIGINAL", model="m"),
        MainIntegratorOutput,
        telemetry=telemetry,
        record_kwargs=_record_kwargs(),
    )
    assert client.last_prompt is not None
    assert "ORIGINAL" in client.last_prompt
    # Schema field names should appear verbatim in the augmented prompt.
    assert "final_answer" in client.last_prompt
    assert "subagent_influence" in client.last_prompt
    assert "JSON Schema" in client.last_prompt
    assert client.last_prompt.index("JSON Schema") < client.last_prompt.index(
        "ORIGINAL"
    )


def test_parse_tolerates_invalid_latex_backslash_escapes():
    """LLMs frequently emit LaTeX (e.g. \\sqrt, \\alpha) inside JSON string
    values. `\\s`, `\\a`, etc. are not valid JSON escapes and would otherwise
    raise `Invalid \\escape`. The robust loader must repair and accept them.
    """
    bad = (
        '{"final_answer":"A","confidence":0.8,'
        '"rationale_summary":"use \\sqrt{x} and \\alpha values",'
        '"subagent_influence":{"A":"not_used","B":"not_used",'
        '"C":"not_used","D":"not_used"}}'
    )
    parsed = parse_json_with_retries(
        lambda _: bad, LLMRequest(prompt="p", model="m"), MainIntegratorOutput
    )
    assert parsed.final_answer == "A"
    assert "sqrt" in parsed.rationale_summary
    assert "alpha" in parsed.rationale_summary


def test_parse_tolerates_prose_around_json_object():
    wrapped = f"Sure, here is the JSON:\n{VALID_JSON}\nLet me know if you need more."
    parsed = parse_json_with_retries(
        lambda _: wrapped, LLMRequest(prompt="p", model="m"), MainIntegratorOutput
    )
    assert parsed.final_answer == "A"


def test_parse_tolerates_trailing_commas():
    bad = (
        '{"final_answer":"A","confidence":0.8,"rationale_summary":"ok",'
        '"subagent_influence":{"A":"not_used","B":"not_used",'
        '"C":"not_used","D":"not_used",},}'
    )
    parsed = parse_json_with_retries(
        lambda _: bad, LLMRequest(prompt="p", model="m"), MainIntegratorOutput
    )
    assert parsed.final_answer == "A"
