from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

from gpqa_cmab.schemas import AgentId

AGENT_IDS: tuple[AgentId, ...] = (AgentId.A, AgentId.B, AgentId.C, AgentId.D)


def normalize_subset(subset: Iterable[AgentId | str]) -> tuple[AgentId, ...]:
    try:
        coerced = [AgentId(item) for item in subset]
    except ValueError as exc:
        raise ValueError(f"Unknown subagent in subset: {exc}") from exc
    values = tuple(sorted(coerced, key=AGENT_IDS.index))
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate subagent in subset: {values}")
    return values


def subset_id(subset: Iterable[AgentId | str]) -> str:
    normalized = normalize_subset(subset)
    return "main_only" if not normalized else ",".join(normalized)


def all_subsets() -> list[tuple[AgentId, ...]]:
    results: list[tuple[AgentId, ...]] = [()]
    for size in range(1, len(AGENT_IDS) + 1):
        results.extend(combinations(AGENT_IDS, size))
    return results
