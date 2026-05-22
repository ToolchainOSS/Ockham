from __future__ import annotations

from typing import Any
from uuid import uuid4

from gpqa_cmab.agents.self_consistency import run_self_consistency
from gpqa_cmab.cost_guard import BudgetExceeded, CostGuard
from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.schemas import GPQAQuestion
from gpqa_cmab.telemetry import TelemetryLogger


def run_self_consistency_experiment(
    questions: list[GPQAQuestion],
    client: LLMClient,
    *,
    model: str,
    k_values: list[int],
    seed: int = 0,
    experiment_id: str | None = None,
    temperature: float = 0.7,
    cost_guard: CostGuard | None = None,
    telemetry: TelemetryLogger | None = None,
) -> list[dict[str, Any]]:
    """Run self-consistency across the dataset for several K values.

    Returns one row per (question, K) with correctness, vote counts, and
    total tokens drawn from the telemetry logger for that batch. Honours an
    optional shared ``CostGuard`` so unbounded SC sweeps (which can easily
    issue ``len(k_values) * sum(k_values) * |questions|`` calls) cannot run
    away from a real provider's bill.
    """
    experiment = experiment_id or f"sc-{uuid4()}"
    guard = cost_guard or CostGuard()
    trace = telemetry
    rows: list[dict[str, Any]] = []
    for question in questions:
        for k in k_values:
            if guard.would_exceed_calls(k) or guard.exhausted():
                return rows
            batch_telemetry = trace or TelemetryLogger()
            start_index = len(batch_telemetry.records)
            try:
                output = run_self_consistency(
                    client,
                    question,
                    k=k,
                    seed=seed,
                    experiment_id=experiment,
                    model=model,
                    telemetry=batch_telemetry,
                    temperature=temperature,
                )
            except BudgetExceeded:
                return rows
            batch_records = batch_telemetry.records[start_index:]
            total_tokens = sum(row.usage.total_tokens for row in batch_records)
            for record in batch_records:
                guard.add_call_usage(record.usage)
            rows.append(
                {
                    "experiment_id": experiment,
                    "question_id": question.question_id,
                    "domain": question.domain,
                    "policy": f"SC-{k}" if k > 1 else "CoT-1",
                    "k": k,
                    "final_answer": output.final_answer,
                    "correct_answer": question.correct_answer,
                    "correct": output.final_answer == question.correct_answer,
                    "confidence": output.confidence,
                    "total_tokens": total_tokens,
                    "num_calls": len(batch_records),
                }
            )
    return rows
