"""Empirical utility data harvested from the GPQA-Diamond Physics MVP.

These numbers are the per-subset ``utility`` column from
``artifacts/results/metrics_summary.json``. They power the offline
``SubagentEnvironment`` reward so the Phase-1 prototype can train without
touching real LLM calls. The set ``frozenset()`` denotes ``main_only`` —
the main integrator running with no helper subagents.
"""

from __future__ import annotations

from collections.abc import Iterable

TOOLS: tuple[str, ...] = ("A", "B", "C", "D")

# Source: artifacts/results/metrics_summary.json (n=86 GPQA-Diamond Physics).
# Kept here as a plain dict so the GFN module never has to read disk.
EMPIRICAL_UTILITIES: dict[frozenset[str], float] = {
    frozenset(): 0.436,  # main_only
    frozenset({"A"}): 0.743,
    frozenset({"B"}): 0.542,
    frozenset({"C"}): 0.797,
    frozenset({"D"}): 0.568,
    frozenset({"A", "B"}): 0.720,
    frozenset({"A", "C"}): 0.801,
    frozenset({"A", "D"}): 0.711,
    frozenset({"B", "C"}): 0.763,
    frozenset({"B", "D"}): 0.581,
    frozenset({"C", "D"}): 0.708,
    frozenset({"A", "B", "C"}): 0.767,
    frozenset({"A", "B", "D"}): 0.677,
    frozenset({"A", "C", "D"}): 0.781,
    frozenset({"B", "C", "D"}): 0.720,
    frozenset({"A", "B", "C", "D"}): 0.736,
}


def subset_to_id(subset: Iterable[str]) -> str:
    """Render a subset as the canonical ``A,B,C`` / ``main_only`` id."""
    ordered = tuple(t for t in TOOLS if t in set(subset))
    return "main_only" if not ordered else ",".join(ordered)


def utilities_table(
    utilities: dict[frozenset[str], float] | None = None,
) -> list[dict[str, float | str]]:
    """Sorted-by-utility view of the empirical reward table (for reports)."""
    src = utilities if utilities is not None else EMPIRICAL_UTILITIES
    rows = [{"subset_id": subset_to_id(k), "utility": v} for k, v in src.items()]
    rows.sort(key=lambda r: r["utility"], reverse=True)
    return rows
