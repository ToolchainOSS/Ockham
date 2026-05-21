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
    target_size: float | None = None,
    seeds: int = 100,
    lambda_token: float = 0.05,
    lambda_call: float = 0.01,
) -> dict[str, float | int | str]:
    """Random budget-matched pruning baseline.

    For each seed and each question, sample a subset uniformly at random
    from subsets of size equal to the target. If `target_size` is provided
    (typically the CMAB's empirical average subset size from replay), it
    overrides the size derived from `target_subset_id` and is rounded to
    the nearest integer in `[0, 4]`.
    """
    by_question_subset: dict[str, dict[str, FactorialResult]] = defaultdict(dict)
    for row in rows:
        by_question_subset[row.question_id][row.subset_id] = row
    all_subset_ids = sorted({row.subset_id for row in rows})
    all_four_rows = [r for r in rows if r.subset_id == "A,B,C,D"]
    all_four_tokens = average_tokens(all_four_rows) or 1.0
    if target_size is None:
        target_rows = [r for r in rows if r.subset_id == target_subset_id]
        derived_size: float = (
            float(len(target_rows[0].selected_subagents)) if target_rows else 4.0
        )
    else:
        derived_size = float(target_size)
    rounded_size = max(0, min(4, round(derived_size)))
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
            candidates = subsets_by_size.get(rounded_size) or list(available)
            sid = rng.choice(candidates)
            if sid not in available:
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
        "target_size": rounded_size,
        "target_size_raw": derived_size,
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
    random_target_size: float | None = None,
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
            target_size=random_target_size,
            seeds=seeds,
            lambda_token=lambda_token,
            lambda_call=lambda_call,
        ),
        "oracle_fixed_subset": oracle_fixed_subset(
            rows, lambda_token=lambda_token, lambda_call=lambda_call
        ),
    }


def cmab_correctness_by_question(
    replay_rows: list[dict],
) -> dict[str, float]:
    """Average correctness per question across all replay (seed, step) rows."""
    sums: dict[str, list[float]] = defaultdict(list)
    for row in replay_rows:
        sums[row["question_id"]].append(float(bool(row["correct"])))
    return {qid: mean(values) for qid, values in sums.items()}


def cmab_avg_subset_size(replay_rows: list[dict]) -> float:
    """Average number of subagents per replay step (across all seeds/steps)."""
    sizes: list[int] = []
    for row in replay_rows:
        sid = row["selected_subset_id"]
        sizes.append(0 if sid == "main_only" else len(sid.split(",")))
    return mean(sizes) if sizes else 0.0


def cmab_avg_tokens(replay_rows: list[dict]) -> float:
    return (
        mean(float(row["total_tokens"]) for row in replay_rows) if replay_rows else 0.0
    )


def _correctness_by_question(rows: list[FactorialResult]) -> dict[str, float]:
    return {row.question_id: float(row.correct) for row in rows}


def _bootstrap_paired_difference(
    a: dict[str, float],
    b: dict[str, float],
    *,
    seed: int = 0,
    samples: int = 1000,
) -> tuple[float, float, float]:
    """Bootstrap a paired mean difference (a - b) over shared question IDs.

    Returns ``(mean_diff, ci_low, ci_high)``.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return (0.0, 0.0, 0.0)
    diffs = [a[q] - b[q] for q in shared]
    rng = random.Random(seed)
    estimates: list[float] = []
    n = len(diffs)
    for _ in range(samples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    lo = estimates[int(0.025 * samples)]
    hi = estimates[int(0.975 * samples) - 1]
    return (sum(diffs) / n, lo, hi)


def non_inferiority_report(
    full_factorial: list[FactorialResult],
    replay_rows: list[dict],
    *,
    epsilon: float = 0.03,
    seed: int = 0,
    samples: int = 1000,
    threshold: float = 0.5,
) -> dict[str, object]:
    """CMAB vs all-four non-inferiority with bootstrap CI and McNemar counts.

    The CMAB's per-question correctness is the mean over replay (seed, step)
    rows for that question. For McNemar, that mean is thresholded at
    ``threshold`` (default 0.5) to produce a binary correctness indicator.
    """
    all_four = [row for row in full_factorial if row.subset_id == "A,B,C,D"]
    if not all_four or not replay_rows:
        return {"status": "insufficient_data"}
    all_four_correct = _correctness_by_question(all_four)
    cmab_correct = cmab_correctness_by_question(replay_rows)
    shared = sorted(set(all_four_correct) & set(cmab_correct))
    if not shared:
        return {"status": "no_overlap"}
    acc_all_four = mean(all_four_correct[q] for q in shared)
    acc_cmab = mean(cmab_correct[q] for q in shared)
    gap, lo, hi = _bootstrap_paired_difference(
        all_four_correct, cmab_correct, seed=seed, samples=samples
    )
    # Threshold CMAB ensemble for McNemar
    cmab_binary = {q: 1.0 if v >= threshold else 0.0 for q, v in cmab_correct.items()}
    mcnemar = {
        "all_four_correct_cmab_wrong": sum(
            int(all_four_correct[q] == 1.0 and cmab_binary[q] == 0.0) for q in shared
        ),
        "all_four_wrong_cmab_correct": sum(
            int(all_four_correct[q] == 0.0 and cmab_binary[q] == 1.0) for q in shared
        ),
    }
    return {
        "accuracy_all_four": acc_all_four,
        "accuracy_cmab": acc_cmab,
        "accuracy_gap": gap,
        "accuracy_gap_ci": [lo, hi],
        "epsilon": epsilon,
        "non_inferior": bool(gap <= epsilon),
        "non_inferior_ci_upper": bool(hi <= epsilon),
        "mcnemar": mcnemar,
        "n_questions": len(shared),
        "bootstrap_samples": samples,
        "mcnemar_threshold": threshold,
    }


def self_consistency_summary(
    rows: list[dict],
    *,
    all_four_avg_tokens: float,
    lambda_token: float = 0.05,
    lambda_call: float = 0.01,
) -> list[dict[str, object]]:
    """Aggregate self-consistency results per K.

    `rows` is the JSONL produced by `run_self_consistency_experiment`.
    Utility uses the same cost-aware formula as the rest of the pipeline;
    `num_subagents` is set to 0 because self-consistency does not invoke
    subagents.
    """
    by_k: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_k[int(row["k"])].append(row)
    summary: list[dict[str, object]] = []
    for k in sorted(by_k):
        bucket = by_k[k]
        avg_tokens = mean(float(item["total_tokens"]) for item in bucket)
        acc = mean(float(bool(item["correct"])) for item in bucket)
        normalized = avg_tokens / all_four_avg_tokens if all_four_avg_tokens else 0.0
        # num_subagents=0 for self-consistency (no helper subagents).
        utility = acc - lambda_token * normalized - lambda_call * 0
        summary.append(
            {
                "k": k,
                "label": "CoT-1" if k == 1 else f"SC-{k}",
                "n": len(bucket),
                "accuracy": acc,
                "avg_tokens": avg_tokens,
                "token_savings_vs_all_four": (
                    1 - avg_tokens / all_four_avg_tokens if all_four_avg_tokens else 0.0
                ),
                "utility": utility,
            }
        )
    return summary
