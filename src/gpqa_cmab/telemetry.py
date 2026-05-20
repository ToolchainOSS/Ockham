from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from gpqa_cmab.schemas import AggregateTelemetry, CallTelemetry, LLMResponse


class TelemetryLogger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.records: list[CallTelemetry] = []
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        response: LLMResponse,
        experiment_id: str,
        question_id: str,
        agent_type: str,
        subset_id: str,
        model: str,
        prompt_version: str,
        temperature: float,
        success: bool = True,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> CallTelemetry:
        telemetry = CallTelemetry(
            experiment_id=experiment_id,
            question_id=question_id,
            agent_type=agent_type,
            subset_id=subset_id,
            model=model,
            prompt_version=prompt_version,
            temperature=temperature,
            usage=response.usage,
            latency_ms=response.latency_ms,
            success=success,
            error_type=error_type,
            error_message=error_message,
        )
        self.records.append(telemetry)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(telemetry.model_dump_json() + "\n")
        return telemetry


def aggregate_usage(
    records: Iterable[CallTelemetry],
    *,
    experiment_id: str,
    question_id: str,
    subset_id: str,
    selected_subagents: list[str],
) -> AggregateTelemetry:
    rows = list(records)
    subagent_tokens = {key: 0 for key in "ABCD"}
    main_tokens = 0
    for row in rows:
        if row.agent_type == "main":
            main_tokens += row.usage.total_tokens
        elif row.agent_type in subagent_tokens:
            subagent_tokens[row.agent_type] += row.usage.total_tokens
    return AggregateTelemetry(
        experiment_id=experiment_id,
        question_id=question_id,
        subset_id=subset_id,
        selected_subagents=selected_subagents,
        total_prompt_tokens=sum(row.usage.prompt_tokens for row in rows),
        total_completion_tokens=sum(row.usage.completion_tokens for row in rows),
        total_tokens=sum(row.usage.total_tokens for row in rows),
        main_tokens=main_tokens,
        subagent_tokens=subagent_tokens,
        num_subagent_calls=len(selected_subagents),
        latency_total_ms=sum(row.latency_ms for row in rows),
        estimated_cost_usd=0.0,
    )


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                handle.write(row.model_dump_json() + "\n")
            else:
                handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
