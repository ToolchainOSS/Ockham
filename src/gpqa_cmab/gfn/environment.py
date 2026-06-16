"""Subset-construction MDP used by the GFlowNet.

State space (DAG)
-----------------
* ``S = {0, 1}^4`` — multi-hot membership vector over ``(A, B, C, D)``.
* ``s_0`` is the empty set; terminal states ``x`` are reached by emitting
  the explicit ``TERMINATE`` action.

Action space ``A = {ADD_A, ADD_B, ADD_C, ADD_D, TERMINATE}`` (size 5).

Reward
------
``R(x) = exp(utility(x) / temperature)`` so the target sampling
distribution emphasises high-utility subsets (smaller temperature ⇒
sharper distribution). ``R(x) > 0`` strictly, as required by GFlowNets.

Backward policy
---------------
For the construction MDP every terminal state ``x`` with ``k`` tools has
``k`` forward parents (one per tool that could have been added last). We
adopt the canonical *uniform-parent* backward policy: ``P_B(s | s') =
1 / |parents(s')|``. Because the trajectory's only non-add transition is
the final ``TERMINATE`` (whose single parent is the same state ``x``),
the trajectory-wide backward log-prob simplifies to
``log P_B(τ) = -log(k!)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gpqa_cmab.gfn.empirical import EMPIRICAL_UTILITIES, TOOLS

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class SubagentEnvironment:
    """Deterministic subset-construction environment for the CMAB-GFN.

    The environment owns the empirical reward dictionary and exposes
    *vectorised* helpers operating on ``torch.Tensor`` states so the
    training loop can keep an entire batch on one device. The TERMINATE
    action is always index ``len(TOOLS)`` (i.e. ``4`` here).
    """

    utilities: dict[frozenset[str], float] = field(
        default_factory=lambda: dict(EMPIRICAL_UTILITIES)
    )
    temperature: float = 0.1
    reward_floor: float = 1e-30  # numerical safety for log R

    @property
    def tools(self) -> tuple[str, ...]:
        return TOOLS

    @property
    def n_tools(self) -> int:
        return len(TOOLS)

    @property
    def n_actions(self) -> int:
        return len(TOOLS) + 1  # 4 ADDs + 1 TERMINATE

    @property
    def terminate_action(self) -> int:
        return len(TOOLS)

    # ---- pure-python helpers (used in tests / reports) --------------------
    def subset_from_state(
        self, state_vec: torch.Tensor | list[float]
    ) -> frozenset[str]:
        items = [
            tool for tool, bit in zip(TOOLS, state_vec, strict=True) if float(bit) > 0.5
        ]
        return frozenset(items)

    def scalar_reward(self, subset: frozenset[str]) -> float:
        u = self.utilities.get(subset, 0.0)
        return math.exp(u / self.temperature)

    # ---- vectorised helpers (used inside the training loop) ---------------
    def initial_state(
        self, batch_size: int, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        import torch

        return torch.zeros(batch_size, self.n_tools, device=device)

    def action_mask(
        self,
        state: torch.Tensor,
        active_arms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a ``(B, n_actions)`` bool mask of *valid* actions.

        ``ADD_X`` is valid iff ``X`` is not already in the set **and**
        ``X`` is permitted by the optional CMAB active-arms mask.
        ``TERMINATE`` is always valid (the empty set is a legal terminal
        state, matching the ``main_only`` arm of the existing pipeline).
        """
        import torch

        already_in = state.bool()
        if active_arms is None:
            allowed = torch.ones_like(already_in, dtype=torch.bool)
        else:
            allowed = active_arms.to(device=state.device, dtype=torch.bool)
            if allowed.dim() == 1:
                allowed = allowed.expand_as(already_in)
        add_mask = (~already_in) & allowed  # (B, n_tools)
        term_mask = torch.ones(
            *state.shape[:-1], 1, dtype=torch.bool, device=state.device
        )
        return torch.cat([add_mask, term_mask], dim=-1)

    def reward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute ``R(x)`` for a batch of terminal states.

        Looked up from ``self.utilities`` and pushed through
        ``exp(u / T)``. Missing entries fall back to ``reward_floor``
        (this branch is unreachable for the 4-tool MVP but defended for
        forward-compat with larger pools).
        """
        import torch

        batch_size = state.shape[0]
        rewards = torch.empty(batch_size, device=state.device)
        for i in range(batch_size):
            subset = self.subset_from_state(state[i])
            if subset in self.utilities:
                rewards[i] = math.exp(self.utilities[subset] / self.temperature)
            else:
                rewards[i] = self.reward_floor
        return rewards
