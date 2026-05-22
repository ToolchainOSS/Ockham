from gpqa_cmab.agents.subagents import run_all_subagents
from gpqa_cmab.cost_guard import CostGuard
from gpqa_cmab.dataset import load_questions
from gpqa_cmab.experiments.factorial import load_subagent_cache, run_full_factorial
from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.llm.mock import MockLLMClient
from gpqa_cmab.schemas import LLMRequest, LLMResponse, Usage
from gpqa_cmab.telemetry import TelemetryLogger, read_jsonl


def test_factorial_runner_creates_16_rows(sample_jsonl):
    questions = load_questions(sample_jsonl, "physics")
    rows = run_full_factorial(
        questions, MockLLMClient(), main_model="main", subagent_model="sub"
    )
    assert len(rows) == 16
    assert {row.subset_id for row in rows}
    assert all(row.correct for row in rows)


def test_factorial_runner_can_persist_full_call_trace(sample_jsonl, tmp_path):
    questions = load_questions(sample_jsonl, "physics")
    trace_path = tmp_path / "factorial_trace.jsonl"
    telemetry = TelemetryLogger(trace_path)
    rows = run_full_factorial(
        questions,
        MockLLMClient(),
        main_model="main",
        subagent_model="sub",
        telemetry=telemetry,
    )

    trace_rows = read_jsonl(trace_path)
    assert len(rows) == 16
    assert len(trace_rows) == 20
    assert {row["agent_type"] for row in trace_rows} == {"main", "A", "B", "C", "D"}
    assert all(row["prompt_sha256"] for row in trace_rows)
    assert all(row["response_sha256"] for row in trace_rows)


class _RetryOnceClient(LLMClient):
    def __init__(self) -> None:
        self._mock = MockLLMClient()
        self._failed = False

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self._failed:
            self._failed = True
            return LLMResponse(
                content="not json",
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                latency_ms=1,
            )
        return self._mock.complete(request)


def test_factorial_cost_guard_counts_json_retry_attempts(sample_jsonl):
    questions = load_questions(sample_jsonl, "physics")
    guard = CostGuard()
    telemetry = TelemetryLogger()
    rows = run_full_factorial(
        questions,
        _RetryOnceClient(),
        main_model="main",
        subagent_model="sub",
        cost_guard=guard,
        telemetry=telemetry,
    )

    assert len(rows) == 16
    assert len(telemetry.records) == 21
    assert guard.calls == 21
    assert guard.total_tokens == sum(
        row.usage.total_tokens for row in telemetry.records
    )


def test_factorial_cache_rows_are_included_in_active_trace(sample_jsonl):
    questions = load_questions(sample_jsonl, "physics")
    cache_telemetry = TelemetryLogger()
    reports, telemetry_rows = run_all_subagents(
        MockLLMClient(),
        questions[0],
        experiment_id="cache-build",
        model="sub",
        telemetry=cache_telemetry,
    )
    cache = load_subagent_cache(
        [
            {
                "question_id": questions[0].question_id,
                "agent": agent,
                "report": reports[agent].model_dump(mode="json"),
                "telemetry": telemetry_rows[index].model_dump(mode="json"),
            }
            for index, agent in enumerate("ABCD")
        ]
    )
    run_telemetry = TelemetryLogger()
    guard = CostGuard()

    rows = run_full_factorial(
        questions,
        MockLLMClient(),
        main_model="main",
        subagent_model="sub",
        subagent_cache=cache,
        cost_guard=guard,
        telemetry=run_telemetry,
    )

    assert len(rows) == 16
    assert len(run_telemetry.records) == 20
    assert guard.calls == 16
    cached_rows = [
        row
        for row in run_telemetry.records
        if row.request_metadata.get("telemetry_source") == "subagent_cache"
    ]
    assert {row.agent_type for row in cached_rows} == {"A", "B", "C", "D"}
