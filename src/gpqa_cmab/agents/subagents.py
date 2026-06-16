from __future__ import annotations

from gpqa_cmab.dataset import question_context
from gpqa_cmab.json_utils import build_record_kwargs, complete_validated
from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.prompts import load_prompt, prompt_version
from gpqa_cmab.schemas import (
    AgentId,
    CallTelemetry,
    GPQAQuestion,
    LLMRequest,
    SubagentAReport,
    SubagentBReport,
    SubagentCReport,
    SubagentDReport,
    SubagentReport,
)
from gpqa_cmab.subsets import AGENT_IDS
from gpqa_cmab.telemetry import TelemetryLogger

SCHEMAS: dict[AgentId, type[SubagentReport]] = {
    AgentId.A: SubagentAReport,
    AgentId.B: SubagentBReport,
    AgentId.C: SubagentCReport,
    AgentId.D: SubagentDReport,
}
PROMPTS: dict[AgentId, str] = {
    AgentId.A: "subagent_A_specialist_v1",
    AgentId.B: "subagent_B_reference_v1",
    AgentId.C: "subagent_C_computation_v1",
    AgentId.D: "subagent_D_verifier_v1",
}


def run_subagent(
    client: LLMClient,
    question: GPQAQuestion,
    agent: AgentId | str,
    *,
    experiment_id: str,
    model: str,
    telemetry: TelemetryLogger,
) -> tuple[SubagentReport, CallTelemetry]:
    agent = AgentId(agent)
    prompt_name = PROMPTS[agent]
    prompt = _build_prompt(prompt_name, question)
    request = LLMRequest(
        prompt=prompt,
        model=model,
        metadata={
            "agent_type": agent.value,
            "mock_correct_answer": question.correct_answer,
        },
    )
    record_kwargs = build_record_kwargs(
        experiment_id=experiment_id,
        question_id=question.question_id,
        agent_type=agent.value,
        subset_id=agent.value,
        model=model,
        prompt_version=prompt_version(prompt_name),
        temperature=0.0,
    )
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
    telemetry: TelemetryLogger | None = None,
) -> tuple[dict[AgentId, SubagentReport], list[CallTelemetry]]:
    telemetry = telemetry or TelemetryLogger()
    reports: dict[AgentId, SubagentReport] = {}
    rows: list[CallTelemetry] = []
    for agent in AGENT_IDS:
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
