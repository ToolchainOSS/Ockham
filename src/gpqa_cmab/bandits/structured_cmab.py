from __future__ import annotations

import math
from dataclasses import dataclass, field

from gpqa_cmab.schemas import AgentId
from gpqa_cmab.subsets import AGENT_IDS, all_subsets, subset_id

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

# Index of the bias/intercept feature inside ``FEATURES`` — extracted as a
# constant so the L2 penalty can skip it (we shrink slopes, never the bias).
INTERCEPT_IDX = 0


def features(
    subset_id_value: str, token_cost: float, avg_all_four_tokens: float
) -> list[float]:
    selected = (
        set() if subset_id_value == "main_only" else set(subset_id_value.split(","))
    )
    values = [1.0]
    values.extend(1.0 if agent in selected else 0.0 for agent in AGENT_IDS)
    pairs = (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"))
    values.extend(1.0 if a in selected and b in selected else 0.0 for a, b in pairs)
    values.append(float(len(selected)))
    values.append(token_cost / avg_all_four_tokens if avg_all_four_tokens else 0.0)
    return values


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


@dataclass
class StructuredCMAB:
    """Online logistic CMAB with shared subset features and a UCB bonus.

    History note (2026 refit)
    -------------------------
    The original implementation suffered three coupled cold-start bugs
    that made the policy collapse onto ``main_only`` within ~20 steps:

    1. **Pessimistic initialization.** All weights at 0 means every
       subset predicts σ(0) = 0.5 on step 1, so the score is dominated
       by the negative cost term ``-λ_token · cost / avg_all_four``.
       Because ``main_only`` is by far the cheapest arm, it always wins
       step 1.
    2. **Intercept shrunk by L2.** A single wrong main_only step pulled
       the intercept negative, dragging *every* subset's score down by
       the same amount — main_only's cost advantage persisted.
    3. **Exploration bonus too small.** ``0.1 / √(1+n)`` is the same
       order of magnitude as the cost penalty, so the bandit could not
       overcome the cost bias to revisit expensive arms.

    Fixes (preserving the constructor API):

    * ``prior_accuracy`` warm-starts the intercept to ``logit(p₀)`` so
      initial predictions hover at ~``p₀`` (default 0.7, the all-four
      accuracy band from the MVP). This is equivalent to a soft
      Bayesian prior centred on a known-reasonable accuracy.
    * ``shrink_intercept=False`` excludes the intercept from the L2
      penalty (the standard recipe for online logistic regression).
    * The UCB bonus now uses the canonical ``√(log(1+t)/(1+n))`` growth
      rate with a larger default scale (``0.3``) so unexplored arms
      keep a meaningful score lead even after the cheap arms have been
      visited a handful of times.

    Set ``prior_accuracy=0.5``, ``shrink_intercept=True``,
    ``uncertainty=0.1``, ``bonus_form='inv_sqrt_n'`` to recover the
    pre-fix (legacy-buggy) behaviour for ablation studies.
    """

    lambda_token: float = 0.05
    lambda_call: float = 0.01
    learning_rate: float = 0.2
    l2: float = 0.01
    uncertainty: float = 0.3  # was 0.1 — too small vs the cost penalty
    seed: int = 0
    # ---- new (bug-fix) knobs --------------------------------------------
    prior_accuracy: float = 0.7  # warm-start intercept = logit(prior_accuracy)
    shrink_intercept: bool = False  # don't L2 the bias term
    bonus_form: str = "ucb1"  # "ucb1" → √(log(1+t)/(1+n)); "inv_sqrt_n" legacy
    # ---- learned state --------------------------------------------------
    weights: list[float] = field(default_factory=lambda: [0.0] * len(FEATURES))
    counts: dict[str, int] = field(default_factory=dict)
    total_plays: int = 0

    # NOTE: ``seed`` is retained for API parity with ``SuperArmThompsonSampler``
    # but this learner is deterministic given history (UCB-style bonus), so we
    # do not instantiate an RNG.

    def __post_init__(self) -> None:
        # Warm-start the intercept ONLY if the caller hasn't already
        # supplied non-zero weights (eg. when resuming a checkpoint).
        if all(w == 0.0 for w in self.weights):
            self.weights = list(self.weights)
            self.weights[INTERCEPT_IDX] = _logit(self.prior_accuracy)

    def _bonus(self, sid: str) -> float:
        n = self.counts.get(sid, 0)
        if self.bonus_form == "inv_sqrt_n":  # legacy
            return self.uncertainty / math.sqrt(1 + n)
        # UCB1-style: grows with total plays so unexplored arms stay attractive.
        return self.uncertainty * math.sqrt(math.log(1 + self.total_plays) / (1 + n))

    def select(self, token_costs: dict[str, float], avg_all_four_tokens: float) -> str:
        def score(subset: tuple[AgentId, ...]) -> float:
            sid = subset_id(subset)
            cost = token_costs.get(sid, avg_all_four_tokens)
            phi = features(sid, cost, avg_all_four_tokens)
            prediction = _sigmoid(
                sum(w * x for w, x in zip(self.weights, phi, strict=True))
            )
            return (
                prediction
                - self.lambda_token * (cost / avg_all_four_tokens)
                - self.lambda_call * len(subset)
                + self._bonus(sid)
            )

        best = max(all_subsets(), key=score)
        return subset_id(best)

    def update(
        self, subset: str, correct: bool, token_cost: float, avg_all_four_tokens: float
    ) -> None:
        phi = features(subset, token_cost, avg_all_four_tokens)
        pred = _sigmoid(sum(w * x for w, x in zip(self.weights, phi, strict=True)))
        error = float(correct) - pred
        new_weights: list[float] = []
        for i, (weight, value) in enumerate(zip(self.weights, phi, strict=True)):
            penalty = (
                self.l2 * weight
                if (self.shrink_intercept or i != INTERCEPT_IDX)
                else 0.0
            )
            new_weights.append(weight + self.learning_rate * (error * value - penalty))
        self.weights = new_weights
        self.counts[subset] = self.counts.get(subset, 0) + 1
        self.total_plays += 1


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))
