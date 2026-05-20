from __future__ import annotations

from itertools import combinations

SUBAGENTS: tuple[str, ...] = ("A", "B", "C", "D")


def normalize_subset(subset: list[str] | tuple[str, ...] | set[str]) -> tuple[str, ...]:
    values = tuple(sorted(subset, key=SUBAGENTS.index))
    unknown = [item for item in values if item not in SUBAGENTS]
    if unknown:
        raise ValueError(f"Unknown subagent(s): {unknown}")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate subagent in subset: {values}")
    return values


def subset_id(subset: list[str] | tuple[str, ...] | set[str]) -> str:
    normalized = normalize_subset(subset)
    return "main_only" if not normalized else ",".join(normalized)


def all_subsets() -> list[tuple[str, ...]]:
    results: list[tuple[str, ...]] = [()]
    for size in range(1, len(SUBAGENTS) + 1):
        results.extend(combinations(SUBAGENTS, size))
    return results
