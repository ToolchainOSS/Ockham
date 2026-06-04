from __future__ import annotations

import random
from dataclasses import dataclass, field

from gpqa_cmab.subsets import all_subsets, subset_id


@dataclass
class SuperArmThompsonSampler:
    """Beta-Bernoulli Thompson sampler over the 16 super-arms.

    History note (2026 refit)
    -------------------------
    The original ``alpha0 = beta0 = 1`` (flat) prior caused
    over-exploration: with ~5 plays per arm per seed, the posterior
    variance stays high and TS draws are still nearly uniform. Together
    with the cost penalty (``-λ_token · cost/avg``) this was enough to
    keep the policy at ~0.76 accuracy and ~4.5 k tokens — worse than
    static-``C`` on both axes.

    Fix: a mildly informative Beta(``alpha0=3``, ``beta0=2``) prior
    (mean 0.6, effective sample size 5) anchors initial picks to a
    plausible accuracy band derived from the MVP factorial. The cost
    penalty then has something meaningful to subtract from. Set
    ``alpha0=1, beta0=1`` to recover the legacy uniform prior for
    ablation.
    """

    lambda_token: float = 0.05
    lambda_call: float = 0.01
    alpha0: float = 3.0  # was 1.0 — mild optimism, ESS=5, mean=0.6
    beta0: float = 2.0  # was 1.0
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
        def score(subset: tuple[str, ...]) -> float:
            sid = subset_id(subset)
            theta = self.rng.betavariate(self.success[sid], self.failure[sid])
            normalized = token_costs.get(sid, avg_all_four_tokens) / avg_all_four_tokens
            return (
                theta - self.lambda_token * normalized - self.lambda_call * len(subset)
            )

        return subset_id(max(all_subsets(), key=score))

    def update(self, subset: str, correct: bool) -> None:
        if correct:
            self.success[subset] += 1
        else:
            self.failure[subset] += 1
