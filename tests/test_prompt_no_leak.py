"""Regression tests: the prompt sent to LLMs must not leak the correct answer.

The mock client gets the answer via `LLMRequest.metadata["mock_correct_answer"]`
so the prompt string itself stays clean. This matters because the same prompt
string is what gets sent to billable real LLMs in production.
"""

from __future__ import annotations

from gpqa_cmab.agents.main_integrator import run_main_integrator
from gpqa_cmab.agents.self_consistency import run_self_consistency
from gpqa_cmab.agents.subagents import run_subagent
from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.llm.mock import MockLLMClient
from gpqa_cmab.schemas import GPQAQuestion, LLMRequest, LLMResponse
from gpqa_cmab.telemetry import TelemetryLogger


class _PromptRecorder(LLMClient):
    """Wraps a real client but records every prompt that flows through it."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.prompts: list[str] = []
        self.metadata: list[dict] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.prompts.append(request.prompt)
        self.metadata.append(dict(request.metadata))
        return self.inner.complete(request)


def _question() -> GPQAQuestion:
    return GPQAQuestion(
        question_id="qx",
        domain="physics",
        question="A toy physics question.",
        choices={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        correct_answer="C",
    )


def _assert_no_leak(prompt: str) -> None:
    assert "MOCK_CORRECT_ANSWER" not in prompt
    # The literal correct-answer letter in a "= C" pattern would also be a leak.
    assert "correct_answer=C" not in prompt
    assert "correct_answer: C" not in prompt


def test_subagent_prompt_does_not_leak_correct_answer():
    recorder = _PromptRecorder(MockLLMClient())
    telemetry = TelemetryLogger()
    report, _ = run_subagent(
        recorder,
        _question(),
        "A",
        experiment_id="t",
        model="mock-sub",
        telemetry=telemetry,
    )
    assert recorder.prompts, "expected at least one LLM call"
    for prompt in recorder.prompts:
        _assert_no_leak(prompt)
    # But the mock client still produced the right answer via metadata.
    assert recorder.metadata[0]["mock_correct_answer"] == "C"
    assert report.recommended_answer == "C"


def test_main_integrator_prompt_does_not_leak_correct_answer():
    recorder = _PromptRecorder(MockLLMClient())
    telemetry = TelemetryLogger()
    sub_report, _ = run_subagent(
        recorder,
        _question(),
        "A",
        experiment_id="t",
        model="mock-sub",
        telemetry=telemetry,
    )
    recorder.prompts.clear()
    recorder.metadata.clear()
    output, _ = run_main_integrator(
        recorder,
        _question(),
        {"A": sub_report},
        experiment_id="t",
        model="mock-main",
        telemetry=telemetry,
    )
    assert recorder.prompts
    for prompt in recorder.prompts:
        _assert_no_leak(prompt)
    assert recorder.metadata[0]["mock_correct_answer"] == "C"
    assert output.final_answer == "C"


def test_self_consistency_prompt_does_not_leak_correct_answer():
    recorder = _PromptRecorder(MockLLMClient())
    telemetry = TelemetryLogger()
    output = run_self_consistency(
        recorder,
        _question(),
        k=2,
        seed=0,
        experiment_id="t",
        model="mock-sc",
        telemetry=telemetry,
    )
    assert len(recorder.prompts) == 2
    for prompt in recorder.prompts:
        _assert_no_leak(prompt)
    assert output.final_answer == "C"
