import hashlib
import json

from gpqa_cmab.schemas import LLMRequest, LLMResponse, Usage
from gpqa_cmab.telemetry import (
    TelemetryLogger,
    aggregate_usage,
    redact_known_secrets,
    write_run_manifest,
)


def test_telemetry_records_and_aggregates_usage():
    logger = TelemetryLogger()
    response = LLMResponse(
        content="{}",
        usage=Usage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        latency_ms=10,
    )
    main = logger.record(
        response=response,
        experiment_id="exp",
        question_id="q",
        agent_type="main",
        subset_id="A",
        model="m",
        prompt_version="p",
        temperature=0.0,
    )
    agent = logger.record(
        response=response,
        experiment_id="exp",
        question_id="q",
        agent_type="A",
        subset_id="A",
        model="m",
        prompt_version="p",
        temperature=0.0,
    )
    agg = aggregate_usage(
        [main, agent],
        experiment_id="exp",
        question_id="q",
        subset_id="A",
        selected_subagents=["A"],
    )
    assert agg.total_tokens == 10
    assert agg.main_tokens == 5
    assert agg.subagent_tokens["A"] == 5


def test_telemetry_records_trace_hashes_and_attempt(tmp_path):
    path = tmp_path / "trace.jsonl"
    logger = TelemetryLogger(path)
    request = LLMRequest(
        prompt="What is measured?",
        model="m",
        metadata={"agent_type": "main", "mock_correct_answer": "A"},
    )
    response = LLMResponse(
        content='{"final_answer":"A"}',
        usage=Usage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        latency_ms=10,
        raw_response={"id": "resp-1"},
    )
    row = logger.record(
        response=response,
        request=request,
        attempt=2,
        schema_name="MainIntegratorOutput",
        experiment_id="exp",
        question_id="q",
        agent_type="main",
        subset_id="A",
        model="m",
        prompt_version="p",
        temperature=0.0,
    )

    assert row.attempt == 2
    assert row.prompt_sha256 == hashlib.sha256(request.prompt.encode()).hexdigest()
    assert row.response_sha256 == hashlib.sha256(response.content.encode()).hexdigest()
    assert row.prompt_chars == len(request.prompt)
    assert row.response_chars == len(response.content)
    assert row.schema_name == "MainIntegratorOutput"
    assert row.request_metadata_keys == ["agent_type", "mock_correct_answer"]
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["request_id"] == row.request_id
    assert persisted["raw_response_sha256"] is not None


def test_write_run_manifest_hashes_inputs_artifacts_and_traces(tmp_path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    input_path.write_text('{"x":1}\n', encoding="utf-8")
    output_path.write_text('{"ok":true}\n', encoding="utf-8")
    trace_path.write_text(
        json.dumps(
            {
                "experiment_id": "exp",
                "question_id": "q",
                "agent_type": "main",
                "subset_id": "main_only",
                "model": "m",
                "prompt_version": "p",
                "temperature": 0.0,
                "attempt": 1,
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                    "estimated": False,
                    "reasoning_tokens": 0,
                },
                "latency_ms": 10,
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    manifest_path = tmp_path / "manifest.json"

    write_run_manifest(
        manifest_path,
        command="unit-test",
        argv=["gpqa-cmab", "unit-test"],
        started_utc="2026-05-21T00:00:00+00:00",
        status="completed",
        inputs=[input_path],
        artifacts=[output_path],
        traces=[trace_path],
        settings={"provider": "mock"},
        budget={"calls": 1},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["command"] == "unit-test"
    assert payload["argv"] == ["gpqa-cmab", "unit-test"]
    assert payload["inputs"][0]["sha256"]
    assert payload["artifacts"][0]["path"] == str(output_path)
    assert payload["traces"][0]["jsonl_rows"] == 1
    assert payload["trace_summary"]["call_rows"] == 1
    assert payload["trace_summary"]["usage"]["total_tokens"] == 5
    assert payload["environment"]["OPENAI_API_KEY"]["redacted"] is True
    assert "sk-test-secret" not in manifest_path.read_text(encoding="utf-8")
    assert payload["settings"] == {"provider": "mock"}
    assert payload["budget"] == {"calls": 1}


def test_error_messages_redact_known_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-private-value")
    assert (
        redact_known_secrets("failed with sk-private-value") == "failed with [REDACTED]"
    )
