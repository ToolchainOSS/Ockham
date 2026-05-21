from __future__ import annotations

from gpqa_cmab.dataset import question_context
from gpqa_cmab.json_utils import complete_validated
from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.prompts import load_prompt, prompt_version
from gpqa_cmab.schemas import (
    CallTelemetry,
    GPQAQuestion,
    LLMRequest,
    SubagentAReport,
    SubagentBReport,
    SubagentCReport,
    SubagentDReport,
    SubagentReport,
)
from gpqa_cmab.telemetry import TelemetryLogger

SCHEMAS = {
    "A": SubagentAReport,
    "B": SubagentBReport,
    "C": SubagentCReport,
    "D": SubagentDReport,
}
PROMPTS = {
    "A": "subagent_A_specialist_v1",
    "B": "subagent_B_reference_v1",
    "C": "subagent_C_computation_v1",
    "D": "subagent_D_verifier_v1",
}


def run_subagent(
    client: LLMClient,
    question: GPQAQuestion,
    agent: str,
    *,
    experiment_id: str,
    model: str,
    telemetry: TelemetryLogger,
) -> tuple[SubagentReport, CallTelemetry]:
    prompt_name = PROMPTS[agent]
    prompt = _build_prompt(prompt_name, question)
    request = LLMRequest(
        prompt=prompt,
        model=model,
        metadata={
            "agent_type": agent,
            "mock_correct_answer": question.correct_answer,
        },
    )
    record_kwargs = {
        "experiment_id": experiment_id,
        "question_id": question.question_id,
        "agent_type": agent,
        "subset_id": agent,
        "model": model,
        "prompt_version": prompt_version(prompt_name),
        "temperature": 0.0,
    }
    parsed, row = complete_validated(
        client,
        request,
        SCHEMAS[agent],
        telemetry=telemetry,
        record_kwargs=record_kwargs,
    )
    return parsed, row


def run_all_subagents(
    client: LLMClient,
    question: GPQAQuestion,
    *,
    experiment_id: str,
    model: str,
) -> tuple[dict[str, SubagentReport], list[CallTelemetry]]:
    telemetry = TelemetryLogger()
    reports: dict[str, SubagentReport] = {}
    rows: list[CallTelemetry] = []
    for agent in "ABCD":
        report, row = run_subagent(
            client,
            question,
            agent,
            experiment_id=experiment_id,
            model=model,
            telemetry=telemetry,
        )
        reports[agent] = report
        rows.append(row)
    return reports, rows


def _build_prompt(prompt_name: str, question: GPQAQuestion) -> str:
    return f"{load_prompt(prompt_name)}\n\n{question_context(question)}"
