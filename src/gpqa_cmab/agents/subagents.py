from __future__ import annotations

import json

from gpqa_cmab.dataset import question_context
from gpqa_cmab.prompts import load_prompt, prompt_version
from gpqa_cmab.schemas import (
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
    client,
    question: GPQAQuestion,
    agent: str,
    *,
    experiment_id: str,
    model: str,
    telemetry: TelemetryLogger,
) -> tuple[SubagentReport, object]:
    prompt_name = PROMPTS[agent]
    prompt = _build_prompt(prompt_name, question)
    request = LLMRequest(prompt=prompt, model=model, metadata={"agent_type": agent})
    response = client.complete(request)
    telemetry_row = telemetry.record(
        response=response,
        experiment_id=experiment_id,
        question_id=question.question_id,
        agent_type=agent,
        subset_id=agent,
        model=model,
        prompt_version=prompt_version(prompt_name),
        temperature=0.0,
    )
    return SCHEMAS[agent].model_validate(json.loads(response.content)), telemetry_row


def run_all_subagents(
    client, question: GPQAQuestion, *, experiment_id: str, model: str
):
    telemetry = TelemetryLogger()
    reports = {}
    rows = []
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
    return (
        f"{load_prompt(prompt_name)}\n\n{question_context(question)}\n\n"
        f"MOCK_CORRECT_ANSWER={question.correct_answer}"
    )
