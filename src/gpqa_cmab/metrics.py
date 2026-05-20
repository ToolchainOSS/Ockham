from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean

from gpqa_cmab.schemas import FactorialResult


def accuracy(rows: list[FactorialResult]) -> float:
    return mean(row.correct for row in rows) if rows else 0.0


def average_tokens(rows: list[FactorialResult]) -> float:
    return mean(row.usage.total_tokens for row in rows) if rows else 0.0


def cost_per_correct(rows: list[FactorialResult]) -> float:
    correct = sum(row.correct for row in rows)
    return (
        sum(row.usage.total_tokens for row in rows) / correct
        if correct
        else float("inf")
    )


def utility(
    row: FactorialResult,
    avg_all_four_tokens: float,
    lambda_token: float,
    lambda_call: float,
) -> float:
    normalized = (
        row.usage.total_tokens / avg_all_four_tokens if avg_all_four_tokens else 0.0
    )
    return (
        float(row.correct)
        - lambda_token * normalized
        - lambda_call * len(row.selected_subagents)
    )


def subset_table(
    rows: list[FactorialResult], lambda_token: float = 0.05, lambda_call: float = 0.01
) -> list[dict]:
    by_subset: dict[str, list[FactorialResult]] = defaultdict(list)
    for row in rows:
        by_subset[row.subset_id].append(row)
    all_four_tokens = average_tokens(by_subset.get("A,B,C,D", [])) or 1.0
    table = []
    for subset_id, subset_rows in sorted(by_subset.items()):
        avg_tokens = average_tokens(subset_rows)
        table.append(
            {
                "subset_id": subset_id,
                "n": len(subset_rows),
                "accuracy": accuracy(subset_rows),
                "avg_tokens": avg_tokens,
                "token_savings_vs_all_four": 1 - avg_tokens / all_four_tokens,
                "cost_per_correct_tokens": cost_per_correct(subset_rows),
                "utility": mean(
                    utility(row, all_four_tokens, lambda_token, lambda_call)
                    for row in subset_rows
                ),
            }
        )
    return table


def bootstrap_ci(
    values: list[float], *, seed: int = 0, samples: int = 1000
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(mean(rng.choice(values) for _ in values))
    estimates.sort()
    return (estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1])


def mcnemar_counts(
    a: list[FactorialResult], b: list[FactorialResult]
) -> dict[str, int]:
    by_a = {row.question_id: row.correct for row in a}
    by_b = {row.question_id: row.correct for row in b}
    both = set(by_a) & set(by_b)
    return {
        "a_correct_b_wrong": sum(by_a[q] and not by_b[q] for q in both),
        "a_wrong_b_correct": sum((not by_a[q]) and by_b[q] for q in both),
    }
