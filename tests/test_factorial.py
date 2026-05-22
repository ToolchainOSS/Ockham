from gpqa_cmab.dataset import load_questions
from gpqa_cmab.experiments.factorial import run_full_factorial
from gpqa_cmab.llm.mock import MockLLMClient
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
