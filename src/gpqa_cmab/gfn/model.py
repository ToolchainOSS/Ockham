"""GFlowNet policy network used by the CMAB-GFN explorer.

The model is a tiny feed-forward MLP mapping a multi-hot state vector
``s ∈ {0, 1}^4`` to ``n_actions = 5`` logits (4 ``ADD_X`` actions + 1
``TERMINATE``). The learnable scalar ``log_Z`` is stored alongside the
network so a single optimizer step updates both per the Trajectory
Balance objective.

Action masking (``CMAB pre-filter`` and ``already-in-set``) is applied
by *replacing* the masked logits with ``-inf`` before the
``log_softmax``, so the resulting distribution is exactly zero on
illegal actions while staying differentiable on the legal ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GFlowNet(nn.Module):
    """Forward policy ``P_F(a | s; θ)`` plus the learnable scalar ``log Z``.

    The network is small on purpose — the action space is 5 and the
    state vector is 4-dimensional, so a wider net would just overfit
    the empirical reward table and hide the diversity-preserving
    behaviour we want to demonstrate.
    """

    def __init__(
        self,
        n_tools: int = 4,
        n_actions: int = 5,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_tools = n_tools
        self.n_actions = n_actions
        self.policy_net = nn.Sequential(
            nn.Linear(n_tools, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        # log Z is a single learnable scalar; parameterising in log-space
        # keeps Z strictly positive and the TB loss numerically stable.
        self.log_z = nn.Parameter(torch.zeros(1))

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.policy_net(state)

    def log_policy(
        self, state: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return masked ``log P_F(· | s)`` of shape ``(B, n_actions)``.

        Illegal actions receive a large negative bias and therefore drop
        to zero probability after the softmax. We use ``-1e9`` instead
        of ``-inf`` so the gradient through unused entries stays finite.
        """
        logits = self.logits(state)
        if action_mask.dtype != torch.bool:
            action_mask = action_mask.bool()
        masked = logits.masked_fill(~action_mask, -1e9)
        return torch.log_softmax(masked, dim=-1)
