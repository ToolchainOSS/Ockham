from __future__ import annotations

from typing import Any
from uuid import uuid4

from gpqa_cmab.agents.main_integrator import run_main_integrator
from gpqa_cmab.agents.subagents import SCHEMAS, run_all_subagents
from gpqa_cmab.cost_guard import BudgetExceeded, CostGuard
from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.prompts import prompt_version
from gpqa_cmab.schemas import (
    CallTelemetry,
    FactorialResult,
    GPQAQuestion,
    SubagentReport,
)
from gpqa_cmab.subsets import all_subsets, subset_id
from gpqa_cmab.telemetry import TelemetryLogger, aggregate_usage

SubagentCache = dict[str, dict[str, dict[str, Any]]]


def load_subagent_cache(rows: list[dict[str, Any]]) -> SubagentCache:
    """Build a `{question_id: {agent: {report, telemetry}}}` index from JSONL.

    Accepts the row shape produced by `gpqa-cmab run-subagents`.
    """
    cache: SubagentCache = {}
    for row in rows:
        question_id = row["question_id"]
        agent = row["agent"]
        cache.setdefault(question_id, {})[agent] = {
            "report": row["report"],
            "telemetry": row.get("telemetry"),
        }
    return cache


def _rehydrate_cached(
    question_id: str,
    cache_entry: dict[str, dict[str, Any]],
) -> tuple[dict[str, SubagentReport], list[CallTelemetry]]:
    reports: dict[str, SubagentReport] = {}
    rows: list[CallTelemetry] = []
    for agent in "ABCD":
        entry = cache_entry.get(agent)
        if entry is None:
            raise KeyError(
                f"Subagent cache missing agent {agent!r} for question {question_id!r}."
            )
        reports[agent] = SCHEMAS[agent].model_validate(entry["report"])
        telemetry_payload = entry.get("telemetry")
        if telemetry_payload is None:
            raise KeyError(
                f"Subagent cache missing telemetry for {agent!r} on {question_id!r}."
            )
        rows.append(CallTelemetry.model_validate(telemetry_payload))
    return reports, rows


def run_full_factorial(
    questions: list[GPQAQuestion],
    client: LLMClient,
    *,
    main_model: str,
    subagent_model: str,
    experiment_id: str | None = None,
    max_api_calls: int | None = None,
    max_estimated_cost_usd: float | None = None,
    cost_usd_per_1k_tokens: float = 0.0,
    subagent_cache: SubagentCache | None = None,
    cost_guard: CostGuard | None = None,
    telemetry: TelemetryLogger | None = None,
) -> list[FactorialResult]:
    """Run the 16-subset factorial sweep.

    Optional `subagent_cache` reuses pre-recorded subagent reports and avoids
    re-running A/B/C/D for every question. Guardrails are enforced at the
    question boundary so that the factorial matrix is always complete for
    every emitted question.

    Pass an external ``cost_guard`` to share a single budget across multiple
    commands (e.g. a global ``MAX_TOTAL_COST_USD`` ceiling). When omitted, a
    local guard is built from the legacy ``max_api_calls`` /
    ``max_estimated_cost_usd`` / ``cost_usd_per_1k_tokens`` arguments.
    """
    experiment = experiment_id or f"exp-{uuid4()}"
    if cost_guard is None:
        cost_guard = CostGuard(
            max_api_calls=max_api_calls,
            max_estimated_cost_usd=max_estimated_cost_usd,
            cost_usd_per_1k_tokens=cost_usd_per_1k_tokens,
        )
    trace = telemetry or TelemetryLogger()
    results: list[FactorialResult] = []
    for question in questions:
        # Refuse to start a question we cannot finish atomically.
        cache_entry = (
            subagent_cache.get(question.question_id) if subagent_cache else None
        )
        planned_subagent_calls = 0 if cache_entry else 4
        planned_calls = planned_subagent_calls + 16  # 16 main-integrator calls.
        if cost_guard.would_exceed_calls(planned_calls):
            break
        if cost_guard.exhausted():
            break

        if cache_entry is not None:
            reports, subagent_rows = _rehydrate_cached(
                question.question_id, cache_entry
            )
            for row in subagent_rows:
                trace.append(
                    row.model_copy(
                        update={
                            "experiment_id": experiment,
                            "request_metadata": {
                                **row.request_metadata,
                                "telemetry_source": "subagent_cache",
                            },
                            "request_metadata_keys": sorted(
                                {*row.request_metadata_keys, "telemetry_source"}
                            ),
                        }
                    )
                )
        else:
            try:
                start_index = len(trace.records)
                reports, subagent_rows = run_all_subagents(
                    client,
                    question,
                    experiment_id=experiment,
                    model=subagent_model,
                    telemetry=trace,
                )
            except BudgetExceeded:
                break
            for row in trace.records_since(start_index):
                cost_guard.add_call(row.usage.total_tokens)

        try:
            for subset in all_subsets():
                selected_reports = {agent: reports[agent] for agent in subset}
                start_index = len(trace.records)
                output, main_row = run_main_integrator(
                    client,
                    question,
                    selected_reports,
                    experiment_id=experiment,
                    model=main_model,
                    telemetry=trace,
                )
                for row in trace.records_since(start_index):
                    cost_guard.add_call(row.usage.total_tokens)
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
        except BudgetExceeded:
            # Drop the partial-matrix rows for this question so consumers
            # never see an incomplete 16-subset block.
            results = [r for r in results if r.question_id != question.question_id]
            break
        # Re-check budget after the question's full block.
        if cost_guard.exhausted():
            break
    return results
