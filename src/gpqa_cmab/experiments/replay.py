from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean

from gpqa_cmab.bandits.structured_cmab import StructuredCMAB
from gpqa_cmab.bandits.superarm_ts import SuperArmThompsonSampler
from gpqa_cmab.metrics import utility
from gpqa_cmab.schemas import BanditStep, FactorialResult


def replay_bandit(
    rows: list[FactorialResult],
    *,
    policy: str,
    seeds: int = 10,
    lambda_token: float = 0.05,
    lambda_call: float = 0.01,
) -> list[BanditStep]:
    by_question: dict[str, dict[str, FactorialResult]] = defaultdict(dict)
    by_subset: dict[str, list[FactorialResult]] = defaultdict(list)
    for row in rows:
        by_question[row.question_id][row.subset_id] = row
        by_subset[row.subset_id].append(row)
    token_costs = {
        sid: mean(row.usage.total_tokens for row in subset_rows)
        for sid, subset_rows in by_subset.items()
    }
    avg_all_four_tokens = token_costs.get(
        "A,B,C,D", max(token_costs.values(), default=1.0)
    )
    steps: list[BanditStep] = []
    question_ids = sorted(by_question)
    for seed in range(seeds):
        rng = random.Random(seed)
        ordered = question_ids[:]
        rng.shuffle(ordered)
        learner = _make_policy(policy, seed, lambda_token, lambda_call)
        cumulative = 0.0
        explored: set[str] = set()
        for index, question_id in enumerate(ordered, start=1):
            selected = learner.select(token_costs, avg_all_four_tokens)
            observed = by_question[question_id][selected]
            observed_utility = utility(
                observed, avg_all_four_tokens, lambda_token, lambda_call
            )
            if isinstance(learner, StructuredCMAB):
                learner.update(
                    selected,
                    observed.correct,
                    observed.usage.total_tokens,
                    avg_all_four_tokens,
                )
            else:
                learner.update(selected, observed.correct)
            cumulative += observed_utility
            explored.add(selected)
            steps.append(
                BanditStep(
                    seed=seed,
                    step=index,
                    policy=policy,
                    question_id=question_id,
                    selected_subset_id=selected,
                    correct=observed.correct,
                    total_tokens=observed.usage.total_tokens,
                    utility=observed_utility,
                    cumulative_utility=cumulative,
                    unique_subsets_explored=len(explored),
                )
            )
    return steps


def _make_policy(
    policy: str, seed: int, lambda_token: float, lambda_call: float
) -> StructuredCMAB | SuperArmThompsonSampler:
    if policy == "structured-cmab":
        return StructuredCMAB(
            lambda_token=lambda_token, lambda_call=lambda_call, seed=seed
        )
    if policy == "superarm-ts":
        return SuperArmThompsonSampler(
            lambda_token=lambda_token, lambda_call=lambda_call, seed=seed
        )
    raise ValueError(f"Unsupported policy: {policy}")
