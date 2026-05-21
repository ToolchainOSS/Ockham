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


def _index_by_subset(
    rows: list[FactorialResult],
) -> dict[str, list[FactorialResult]]:
    indexed: dict[str, list[FactorialResult]] = defaultdict(list)
    for row in rows:
        indexed[row.subset_id].append(row)
    return indexed


def static_pruning_baseline(
    rows: list[FactorialResult],
    *,
    subset_id: str = "A",
    lambda_token: float = 0.05,
    lambda_call: float = 0.01,
) -> dict[str, float | int | str]:
    """Evaluate a fixed (static) pruned subset as a non-adaptive baseline."""
    by_subset = _index_by_subset(rows)
    subset_rows = by_subset.get(subset_id, [])
    all_four_tokens = average_tokens(by_subset.get("A,B,C,D", [])) or 1.0
    if not subset_rows:
        return {
            "policy": "static_pruning",
            "subset_id": subset_id,
            "accuracy": 0.0,
            "avg_tokens": 0.0,
            "token_savings_vs_all_four": 0.0,
            "utility": 0.0,
            "n": 0,
        }
    return {
        "policy": "static_pruning",
        "subset_id": subset_id,
        "accuracy": accuracy(subset_rows),
        "avg_tokens": average_tokens(subset_rows),
        "token_savings_vs_all_four": 1 - average_tokens(subset_rows) / all_four_tokens,
        "cost_per_correct_tokens": cost_per_correct(subset_rows),
        "utility": mean(
            utility(row, all_four_tokens, lambda_token, lambda_call)
            for row in subset_rows
        ),
        "n": len(subset_rows),
    }


def random_pruning_baseline(
    rows: list[FactorialResult],
    *,
    target_subset_id: str = "A,B,C,D",
    seeds: int = 100,
    lambda_token: float = 0.05,
    lambda_call: float = 0.01,
) -> dict[str, float | int | str]:
    """Random budget-matched pruning baseline.

    For each seed, for each question, randomly sample one of the 16 subsets
    such that the *average* number of selected subagents matches the target
    subset's size. We achieve that by sampling subsets uniformly at random
    from the set of subsets whose size equals the target's size on
    expectation — concretely we sample sizes uniformly from {0..4} biased
    toward target_size, then sample a subset of that size uniformly.
    """
    by_question_subset: dict[str, dict[str, FactorialResult]] = defaultdict(dict)
    for row in rows:
        by_question_subset[row.question_id][row.subset_id] = row
    all_subset_ids = sorted({row.subset_id for row in rows})
    all_four_rows = [r for r in rows if r.subset_id == "A,B,C,D"]
    all_four_tokens = average_tokens(all_four_rows) or 1.0
    target_rows = [r for r in rows if r.subset_id == target_subset_id]
    target_size = len(target_rows[0].selected_subagents) if target_rows else 4
    # Group available subsets by number of selected subagents.
    subsets_by_size: dict[int, list[str]] = defaultdict(list)
    for sid in all_subset_ids:
        size = 0 if sid == "main_only" else len(sid.split(","))
        subsets_by_size[size].append(sid)

    correctness: list[float] = []
    token_totals: list[float] = []
    utilities: list[float] = []
    for seed in range(seeds):
        rng = random.Random(seed)
        for _question_id, available in by_question_subset.items():
            # Sample a size centered around target_size with a small spread.
            candidate_sizes = [
                s
                for s in subsets_by_size
                if abs(s - target_size) <= 0 or s == target_size
            ] or list(subsets_by_size)
            size = rng.choice(candidate_sizes)
            sid = rng.choice(subsets_by_size[size])
            if sid not in available:
                # Fallback to any available subset.
                sid = rng.choice(list(available))
            observed = available[sid]
            correctness.append(float(observed.correct))
            token_totals.append(float(observed.usage.total_tokens))
            utilities.append(
                utility(observed, all_four_tokens, lambda_token, lambda_call)
            )
    avg_tokens = mean(token_totals) if token_totals else 0.0
    return {
        "policy": "random_budget_matched",
        "target_subset_id": target_subset_id,
        "target_size": target_size,
        "seeds": seeds,
        "accuracy": mean(correctness) if correctness else 0.0,
        "avg_tokens": avg_tokens,
        "token_savings_vs_all_four": (
            1 - avg_tokens / all_four_tokens if all_four_tokens else 0.0
        ),
        "utility": mean(utilities) if utilities else 0.0,
    }


def oracle_fixed_subset(
    rows: list[FactorialResult],
    *,
    lambda_token: float = 0.05,
    lambda_call: float = 0.01,
) -> dict[str, float | int | str]:
    """Oracle fixed-subset reference: best utility after exhaustive eval."""
    table = subset_table(rows, lambda_token=lambda_token, lambda_call=lambda_call)
    if not table:
        return {"policy": "oracle_fixed_subset", "subset_id": "", "utility": 0.0}
    best = max(table, key=lambda entry: entry["utility"])
    return {"policy": "oracle_fixed_subset_mvp_only", **best}


def baseline_summary(
    rows: list[FactorialResult],
    *,
    static_subset_id: str = "A",
    random_target_subset_id: str = "A,B,C,D",
    seeds: int = 100,
    lambda_token: float = 0.05,
    lambda_call: float = 0.01,
) -> dict[str, object]:
    return {
        "static_pruning": static_pruning_baseline(
            rows,
            subset_id=static_subset_id,
            lambda_token=lambda_token,
            lambda_call=lambda_call,
        ),
        "random_budget_matched": random_pruning_baseline(
            rows,
            target_subset_id=random_target_subset_id,
            seeds=seeds,
            lambda_token=lambda_token,
            lambda_call=lambda_call,
        ),
        "oracle_fixed_subset": oracle_fixed_subset(
            rows, lambda_token=lambda_token, lambda_call=lambda_call
        ),
    }
