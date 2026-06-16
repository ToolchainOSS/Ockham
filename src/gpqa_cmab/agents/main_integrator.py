from __future__ import annotations

import json

from gpqa_cmab.dataset import question_context
from gpqa_cmab.json_utils import build_record_kwargs, complete_validated
from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.prompts import load_prompt, prompt_version
from gpqa_cmab.schemas import (
    AgentId,
    CallTelemetry,
    GPQAQuestion,
    LLMRequest,
    MainIntegratorOutput,
    SubagentReport,
)
from gpqa_cmab.subsets import subset_id
from gpqa_cmab.telemetry import TelemetryLogger


def run_main_integrator(
    client: LLMClient,
    question: GPQAQuestion,
    selected_reports: dict[AgentId, SubagentReport] | dict[str, SubagentReport],
    *,
    experiment_id: str,
    model: str,
    telemetry: TelemetryLogger,
) -> tuple[MainIntegratorOutput, CallTelemetry]:
    normalized = {AgentId(key): value for key, value in selected_reports.items()}
    selected = tuple(normalized)
    sid = subset_id(selected)
    prompt_name = "main_integrator_v1"
    report_text = json.dumps(
        {key.value: value.model_dump(mode="json") for key, value in normalized.items()},
        sort_keys=True,
    )
    prompt = (
        f"{load_prompt(prompt_name)}\n\n{question_context(question)}\n\n"
        f"Selected subagent reports JSON:\n{report_text}"
    )
    request = LLMRequest(
        prompt=prompt,
        model=model,
        metadata={
            "agent_type": "main",
            "mock_correct_answer": question.correct_answer,
        },
    )
    record_kwargs = build_record_kwargs(
        experiment_id=experiment_id,
        question_id=question.question_id,
        agent_type="main",
        subset_id=sid,
        model=model,
        prompt_version=prompt_version(prompt_name),
        temperature=0.0,
    )
    return complete_validated(
        client,
        request,
        MainIntegratorOutput,
        telemetry=telemetry,
        record_kwargs=record_kwargs,
    )
