"""CMAB pre-filter that bounds the GFN's combinatorial topology.

The CMAB-GFN architecture relies on a cheap bandit-style summary of
each base arm's expected utility to **prune** arms whose marginal
contribution is below a user-chosen threshold ``γ``. The GFN then
samples only inside the bounded subspace ``S_restricted``, which (per
the literature) drastically lowers the sample complexity of the
diversity-seeking flow network on combinatorial state spaces.

For the 4-tool MVP we do not need an online UCB loop — the empirical
utility dictionary already plays the role of a pre-filtered oracle.
This module exposes the *interface* that future plug-in CMAB updaters
must satisfy:

    >>> mask = CMABFilter.from_single_arm_utility(EMPIRICAL_UTILITIES,
    ...                                          gamma=0.6).mask
    >>> # mask is a torch.BoolTensor of shape (n_tools,)

so that ``SubagentEnvironment.action_mask(..., active_arms=mask)``
zeros out ``P_F`` on the pruned branches at *every* state.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gpqa_cmab.gfn.empirical import EMPIRICAL_UTILITIES, TOOLS

if TYPE_CHECKING:
    import torch


def single_arm_utilities(
    utilities: dict[frozenset[str], float] | None = None,
    tools: Iterable[str] = TOOLS,
) -> dict[str, float]:
    """Return the utility of each *solo* subset ``{tool}``.

    This is the simplest CMAB summary and matches the historical
    "arm-only" UCB statistic: ``U(i) = E[reward | only arm i played]``.
    """
    src = utilities if utilities is not None else EMPIRICAL_UTILITIES
    return {t: float(src.get(frozenset({t}), 0.0)) for t in tools}


def marginal_contributions(
    utilities: dict[frozenset[str], float] | None = None,
    tools: Iterable[str] = TOOLS,
) -> dict[str, float]:
    """Average lift ``E[u | i ∈ S] - E[u | i ∉ S]`` for each arm ``i``.

    This is more robust than the solo-arm view when arms interact (e.g.
    a tool that is mediocre alone but synergises with another). Negative
    values are evidence that the arm hurts on average and is a good
    candidate for CMAB pruning.
    """
    src = utilities if utilities is not None else EMPIRICAL_UTILITIES
    out: dict[str, float] = {}
    for tool in tools:
        with_t = [u for s, u in src.items() if tool in s]
        without_t = [u for s, u in src.items() if tool not in s]
        m_with = sum(with_t) / len(with_t) if with_t else 0.0
        m_without = sum(without_t) / len(without_t) if without_t else 0.0
        out[tool] = m_with - m_without
    return out


@dataclass(frozen=True)
class CMABFilter:
    """A static γ-threshold pre-filter over the base arms.

    ``scores`` holds the per-arm CMAB summary (single-arm utility,
    marginal contribution, future UCB statistic, …). ``active_arms`` is
    the bool list mirroring ``TOOLS`` order — ``True`` means the arm is
    kept in the GFN's bounded subspace.
    """

    scores: dict[str, float]
    gamma: float
    active_arms: tuple[bool, ...]
    method: str  # "single_arm" | "marginal" | "custom"

    @classmethod
    def from_single_arm_utility(
        cls,
        utilities: dict[frozenset[str], float] | None = None,
        *,
        gamma: float = 0.6,
        tools: Iterable[str] = TOOLS,
    ) -> CMABFilter:
        scores = single_arm_utilities(utilities, tools)
        active = tuple(scores[t] >= gamma for t in tools)
        return cls(scores=scores, gamma=gamma, active_arms=active, method="single_arm")

    @classmethod
    def from_marginal(
        cls,
        utilities: dict[frozenset[str], float] | None = None,
        *,
        gamma: float = 0.0,
        tools: Iterable[str] = TOOLS,
    ) -> CMABFilter:
        scores = marginal_contributions(utilities, tools)
        active = tuple(scores[t] >= gamma for t in tools)
        return cls(scores=scores, gamma=gamma, active_arms=active, method="marginal")

    @classmethod
    def all_active(cls, tools: Iterable[str] = TOOLS) -> CMABFilter:
        """No-op filter — keeps every arm. Useful for ablations."""
        scores = {t: float("inf") for t in tools}
        return cls(
            scores=scores,
            gamma=float("-inf"),
            active_arms=tuple(True for _ in tools),
            method="custom",
        )

    @property
    def mask(self) -> torch.Tensor:
        """1-D ``torch.BoolTensor`` shaped ``(n_tools,)``."""
        import torch

        return torch.tensor(self.active_arms, dtype=torch.bool)

    def summary(self) -> dict[str, object]:
        """Serializable summary suitable for run manifests."""
        return {
            "method": self.method,
            "gamma": self.gamma,
            "scores": dict(self.scores),
            "active_arms": list(self.active_arms),
            "active_tools": [
                t for t, on in zip(TOOLS, self.active_arms, strict=True) if on
            ],
        }
