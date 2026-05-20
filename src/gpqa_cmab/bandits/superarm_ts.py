from __future__ import annotations

import random
from dataclasses import dataclass, field

from gpqa_cmab.subsets import all_subsets, subset_id


@dataclass
class SuperArmThompsonSampler:
    lambda_token: float = 0.05
    lambda_call: float = 0.01
    alpha0: float = 1.0
    beta0: float = 1.0
    seed: int = 0
    success: dict[str, float] = field(default_factory=dict)
    failure: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        for subset in all_subsets():
            sid = subset_id(subset)
            self.success.setdefault(sid, self.alpha0)
            self.failure.setdefault(sid, self.beta0)

    def select(self, token_costs: dict[str, float], avg_all_four_tokens: float) -> str:
        best_sid = "main_only"
        best_score = float("-inf")
        for subset in all_subsets():
            sid = subset_id(subset)
            theta = self.rng.betavariate(self.success[sid], self.failure[sid])
            normalized = token_costs.get(sid, avg_all_four_tokens) / avg_all_four_tokens
            score = (
                theta - self.lambda_token * normalized - self.lambda_call * len(subset)
            )
            if score > best_score:
                best_sid, best_score = sid, score
        return best_sid

    def update(self, subset: str, correct: bool) -> None:
        if correct:
            self.success[subset] += 1
        else:
            self.failure[subset] += 1
