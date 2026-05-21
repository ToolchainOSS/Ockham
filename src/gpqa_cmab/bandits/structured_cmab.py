from __future__ import annotations

import math
from dataclasses import dataclass, field

from gpqa_cmab.subsets import SUBAGENTS, all_subsets, subset_id

FEATURES = (
    "intercept",
    "A",
    "B",
    "C",
    "D",
    "A*B",
    "A*C",
    "A*D",
    "B*C",
    "B*D",
    "C*D",
    "num_subagents",
    "estimated_token_cost",
)


def features(
    subset_id_value: str, token_cost: float, avg_all_four_tokens: float
) -> list[float]:
    selected = (
        set() if subset_id_value == "main_only" else set(subset_id_value.split(","))
    )
    values = [1.0]
    values.extend(1.0 if agent in selected else 0.0 for agent in SUBAGENTS)
    pairs = (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"))
    values.extend(1.0 if a in selected and b in selected else 0.0 for a, b in pairs)
    values.append(float(len(selected)))
    values.append(token_cost / avg_all_four_tokens if avg_all_four_tokens else 0.0)
    return values


@dataclass
class StructuredCMAB:
    lambda_token: float = 0.05
    lambda_call: float = 0.01
    learning_rate: float = 0.2
    l2: float = 0.01
    uncertainty: float = 0.1
    seed: int = 0
    weights: list[float] = field(default_factory=lambda: [0.0] * len(FEATURES))
    counts: dict[str, int] = field(default_factory=dict)

    # NOTE: ``seed`` is retained for API parity with ``SuperArmThompsonSampler``
    # but this learner is deterministic given history (UCB-style bonus), so we
    # do not instantiate an RNG.

    def select(self, token_costs: dict[str, float], avg_all_four_tokens: float) -> str:
        def score(subset: tuple[str, ...]) -> float:
            sid = subset_id(subset)
            cost = token_costs.get(sid, avg_all_four_tokens)
            phi = features(sid, cost, avg_all_four_tokens)
            prediction = _sigmoid(
                sum(w * x for w, x in zip(self.weights, phi, strict=True))
            )
            bonus = self.uncertainty / math.sqrt(1 + self.counts.get(sid, 0))
            return (
                prediction
                - self.lambda_token * (cost / avg_all_four_tokens)
                - self.lambda_call * len(subset)
                + bonus
            )

        return subset_id(max(all_subsets(), key=score))

    def update(
        self, subset: str, correct: bool, token_cost: float, avg_all_four_tokens: float
    ) -> None:
        phi = features(subset, token_cost, avg_all_four_tokens)
        pred = _sigmoid(sum(w * x for w, x in zip(self.weights, phi, strict=True)))
        error = float(correct) - pred
        self.weights = [
            weight + self.learning_rate * (error * value - self.l2 * weight)
            for weight, value in zip(self.weights, phi, strict=True)
        ]
        self.counts[subset] = self.counts.get(subset, 0) + 1


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))
