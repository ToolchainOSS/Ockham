"""Usage aggregation and trace summarization."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from gpqa_cmab.schemas import AgentId, AggregateTelemetry, CallTelemetry
from gpqa_cmab.telemetry.io import read_jsonl


def aggregate_usage(
    records: Iterable[CallTelemetry],
    *,
    experiment_id: str,
    question_id: str,
    subset_id: str,
    selected_subagents: list[AgentId],
) -> AggregateTelemetry:
    rows = list(records)
    subagent_tokens: dict[str, int] = dict.fromkeys("ABCD", 0)
    main_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    latency_total_ms = 0
    for row in rows:
        usage = row.usage
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        total_tokens += usage.total_tokens
        latency_total_ms += row.latency_ms
        if row.agent_type == "main":
            main_tokens += usage.total_tokens
        elif row.agent_type in subagent_tokens:
            subagent_tokens[row.agent_type] += usage.total_tokens
    return AggregateTelemetry(
        experiment_id=experiment_id,
        question_id=question_id,
        subset_id=subset_id,
        selected_subagents=selected_subagents,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        main_tokens=main_tokens,
        subagent_tokens=subagent_tokens,
        num_subagent_calls=len(selected_subagents),
        latency_total_ms=latency_total_ms,
        estimated_cost_usd=0.0,
    )


def summarize_traces(paths: Iterable[Path]) -> dict[str, object]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            rows.extend(read_jsonl(path))
    return summarize_trace_rows(rows)


def summarize_trace_rows(
    rows: Iterable[dict[str, Any] | CallTelemetry],
) -> dict[str, object]:
    normalized = [
        row.model_dump(mode="json") if hasattr(row, "model_dump") else row
        for row in rows
    ]
    by_agent: dict[str, int] = {}
    by_model: dict[str, int] = {}
    error_types: dict[str, int] = {}
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    reasoning_tokens = 0
    cached_prompt_tokens = 0
    prompt_audio_tokens = 0
    completion_audio_tokens = 0
    estimated_usage_rows = 0
    successes = 0
    failures = 0
    max_attempt = 0
    for row in normalized:
        agent = str(row.get("agent_type", "unknown"))
        model = str(row.get("model", "unknown"))
        by_agent[agent] = by_agent.get(agent, 0) + 1
        by_model[model] = by_model.get(model, 0) + 1
        usage = row.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
        reasoning_tokens += int(usage.get("reasoning_tokens") or 0)
        cached_prompt_tokens += int(usage.get("cached_prompt_tokens") or 0)
        prompt_audio_tokens += int(usage.get("prompt_audio_tokens") or 0)
        completion_audio_tokens += int(usage.get("completion_audio_tokens") or 0)
        if usage.get("estimated"):
            estimated_usage_rows += 1
        if row.get("success"):
            successes += 1
        else:
            failures += 1
            error_type = str(row.get("error_type") or "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        max_attempt = max(max_attempt, int(row.get("attempt") or 0))
    return {
        "call_rows": len(normalized),
        "successes": successes,
        "failures": failures,
        "max_attempt": max_attempt,
        "by_agent": dict(sorted(by_agent.items())),
        "by_model": dict(sorted(by_model.items())),
        "error_types": dict(sorted(error_types.items())),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "prompt_audio_tokens": prompt_audio_tokens,
            "completion_audio_tokens": completion_audio_tokens,
            "estimated_rows": estimated_usage_rows,
        },
    }
