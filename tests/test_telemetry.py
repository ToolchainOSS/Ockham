from gpqa_cmab.schemas import LLMResponse, Usage
from gpqa_cmab.telemetry import TelemetryLogger, aggregate_usage


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
