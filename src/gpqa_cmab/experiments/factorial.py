from __future__ import annotations

from uuid import uuid4

from gpqa_cmab.agents.main_integrator import run_main_integrator
from gpqa_cmab.agents.subagents import run_all_subagents
from gpqa_cmab.prompts import prompt_version
from gpqa_cmab.schemas import FactorialResult, GPQAQuestion
from gpqa_cmab.subsets import all_subsets, subset_id
from gpqa_cmab.telemetry import TelemetryLogger, aggregate_usage


def run_full_factorial(
    questions: list[GPQAQuestion],
    client,
    *,
    main_model: str,
    subagent_model: str,
    experiment_id: str | None = None,
    max_api_calls: int | None = None,
) -> list[FactorialResult]:
    experiment = experiment_id or f"exp-{uuid4()}"
    results: list[FactorialResult] = []
    calls = 0
    for question in questions:
        reports, subagent_rows = run_all_subagents(
            client, question, experiment_id=experiment, model=subagent_model
        )
        calls += 4
        for subset in all_subsets():
            if max_api_calls is not None and calls >= max_api_calls:
                return results
            selected_reports = {agent: reports[agent] for agent in subset}
            telemetry = TelemetryLogger()
            output, main_row = run_main_integrator(
                client,
                question,
                selected_reports,
                experiment_id=experiment,
                model=main_model,
                telemetry=telemetry,
            )
            calls += 1
            sid = subset_id(subset)
            selected_subagent_rows = [
                row for row in subagent_rows if row.agent_type in subset
            ]
            usage = aggregate_usage(
                [*selected_subagent_rows, main_row],
                experiment_id=experiment,
                question_id=question.question_id,
                subset_id=sid,
                selected_subagents=list(subset),
            )
            results.append(
                FactorialResult(
                    experiment_id=experiment,
                    question_id=question.question_id,
                    domain=question.domain,
                    subset_id=sid,
                    selected_subagents=list(subset),
                    final_answer=output.final_answer,
                    correct_answer=question.correct_answer,
                    correct=output.final_answer == question.correct_answer,
                    confidence=output.confidence,
                    usage=usage,
                    prompt_versions={
                        "main": prompt_version("main_integrator_v1"),
                        "A": prompt_version("subagent_A_specialist_v1"),
                        "B": prompt_version("subagent_B_reference_v1"),
                        "C": prompt_version("subagent_C_computation_v1"),
                        "D": prompt_version("subagent_D_verifier_v1"),
                    },
                )
            )
    return results
