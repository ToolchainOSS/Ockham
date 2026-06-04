"""Trajectory Balance training loop for the CMAB-GFN explorer.

Implements the trajectory-balance objective for the subset-construction
MDP defined in :mod:`gpqa_cmab.gfn.environment`. For a sampled
trajectory ``τ = (s_0, s_1, …, s_n = x)`` the loss is::

    L_TB(τ; θ, Z) = ( log Z
                    + Σ_t log P_F(s_{t+1} | s_t; θ)
                    - log R(x)
                    - Σ_t log P_B(s_t | s_{t+1}) )^2

The backward policy is uniform over forward parents, which collapses to
``Σ_t log P_B = -log(k!)`` where ``k`` is the number of tools in the
terminal subset (see :mod:`gpqa_cmab.gfn.environment` for the derivation).

The CMAB pre-filter mask is forwarded into ``env.action_mask`` so the
GFN cannot place probability on pruned arms — this is the
"CMAB-GFN intersection" point from the design spec.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from gpqa_cmab.gfn.cmab_filter import CMABFilter
from gpqa_cmab.gfn.empirical import TOOLS, subset_to_id
from gpqa_cmab.gfn.environment import SubagentEnvironment
from gpqa_cmab.gfn.model import GFlowNet

if TYPE_CHECKING:
    pass


@dataclass
class TrainingHistory:
    """Per-checkpoint training trace for the run manifest."""

    iters: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    log_z: list[float] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Empirical post-training diagnostics."""

    n_samples: int
    subset_counts: dict[str, int]
    subset_freqs: dict[str, float]
    target_freqs: dict[str, float]
    mode_share_top1: float
    unique_terminals: int
    avg_subset_size: float
    learned_log_z: float


@dataclass
class TrainResult:
    """Output bundle returned by :func:`train_cmab_gfn`."""

    model: GFlowNet
    history: TrainingHistory
    evaluation: EvaluationReport
    cmab_filter: CMABFilter
    config: dict[str, object]


# ---------------------------------------------------------------------------
# Trajectory sampling
# ---------------------------------------------------------------------------
def sample_trajectories(
    model: GFlowNet,
    env: SubagentEnvironment,
    batch_size: int,
    *,
    active_arms: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one batched rollout under the current forward policy.

    Returns the per-trajectory terminal states plus three log quantities
    aligned to the trajectories::

        terminal_states  (B, n_tools)  float
        log_pf_sum       (B,)          ΣlogP_F over sampled actions
        log_pb_sum       (B,)          ΣlogP_B (uniform-parent identity)
        log_reward       (B,)          log R(x) = utility(x) / T

    The rollout runs for at most ``n_tools + 1`` micro-steps because the
    longest legal trajectory adds all four tools and then terminates.
    """
    state = env.initial_state(batch_size, device=device)
    log_pf = torch.zeros(batch_size, device=device)
    n_adds = torch.zeros(batch_size, device=device, dtype=torch.long)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    max_depth = env.n_tools + 1
    for _ in range(max_depth):
        mask = env.action_mask(state, active_arms)
        log_probs = model.log_policy(state, mask)  # (B, n_actions)
        # ``torch.multinomial`` needs probabilities; clamp underflow.
        probs = log_probs.exp()
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        step_log_pf = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        # Already-terminated trajectories contribute nothing to the sum.
        log_pf = log_pf + torch.where(done, torch.zeros_like(step_log_pf), step_log_pf)
        is_terminate = action == env.terminate_action
        is_add = (~is_terminate) & (~done)
        # Apply the ADD by writing 1.0 into the chosen tool slot.
        # ``clamp(max=n_tools-1)`` makes the one-hot legal even when the
        # action equals ``TERMINATE``; we then zero those rows via ``is_add``.
        add_one_hot = F.one_hot(
            action.clamp(max=env.n_tools - 1), num_classes=env.n_tools
        ).to(state.dtype)
        state = torch.where(is_add.unsqueeze(-1), state + add_one_hot, state)
        n_adds = n_adds + is_add.long()
        done = done | is_terminate
        if bool(done.all().item()):
            break

    # Uniform-parent backward log-prob: -log(k!) with k = n_adds.
    log_pb = -torch.lgamma(n_adds.float() + 1.0)
    log_reward = torch.log(env.reward(state).clamp_min(env.reward_floor))
    return state, log_pf, log_pb, log_reward


def trajectory_balance_loss(
    model: GFlowNet,
    env: SubagentEnvironment,
    batch_size: int,
    *,
    active_arms: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Mean Trajectory Balance loss across a batch of fresh rollouts."""
    _, log_pf, log_pb, log_r = sample_trajectories(
        model, env, batch_size, active_arms=active_arms, device=device
    )
    residual = model.log_z + log_pf - log_pb - log_r
    return (residual**2).mean()


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------
def train_cmab_gfn(
    *,
    env: SubagentEnvironment | None = None,
    cmab_filter: CMABFilter | None = None,
    num_iters: int = 2000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    log_z_learning_rate: float = 1e-2,
    hidden_dim: int = 64,
    seed: int = 0,
    eval_samples: int = 1000,
    checkpoint_every: int = 100,
    device: str = "cpu",
    progress_callback=None,
) -> TrainResult:
    """End-to-end CMAB-GFN training + evaluation.

    Parameters
    ----------
    env
        Subset-construction environment. Defaults to the empirical
        GPQA-Diamond Physics reward with ``temperature=0.1``.
    cmab_filter
        Optional CMAB pre-filter. Defaults to the single-arm-utility
        filter with ``γ = 0.6`` (prunes ``B`` and ``D`` for the MVP data,
        which is exactly the marginal-loser pair flagged in the report).
    progress_callback
        Optional ``fn(iter, loss, log_z) -> None`` hook for the CLI.
    """
    env = env or SubagentEnvironment()
    cmab_filter = cmab_filter or CMABFilter.from_single_arm_utility(env.utilities)
    torch_device = torch.device(device)

    torch.manual_seed(seed)
    model = GFlowNet(
        n_tools=env.n_tools, n_actions=env.n_actions, hidden_dim=hidden_dim
    ).to(torch_device)
    # Separate LR for log_Z (it is one scalar carrying a lot of signal and
    # benefits from a more aggressive step than the network weights).
    optimizer = torch.optim.Adam(
        [
            {
                "params": [p for n, p in model.named_parameters() if n != "log_z"],
                "lr": learning_rate,
            },
            {"params": [model.log_z], "lr": log_z_learning_rate},
        ]
    )
    active_mask = cmab_filter.mask.to(torch_device)
    history = TrainingHistory()

    for it in range(num_iters):
        loss = trajectory_balance_loss(
            model, env, batch_size, active_arms=active_mask, device=torch_device
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % checkpoint_every == 0 or it == num_iters - 1:
            history.iters.append(it)
            history.losses.append(float(loss.item()))
            history.log_z.append(float(model.log_z.item()))
            if progress_callback is not None:
                progress_callback(it, float(loss.item()), float(model.log_z.item()))

    evaluation = evaluate_policy(
        model,
        env,
        n_samples=eval_samples,
        active_arms=active_mask,
        device=torch_device,
    )
    config = {
        "num_iters": num_iters,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "log_z_learning_rate": log_z_learning_rate,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "eval_samples": eval_samples,
        "temperature": env.temperature,
        "device": str(torch_device),
        "cmab_filter": cmab_filter.summary(),
    }
    return TrainResult(
        model=model,
        history=history,
        evaluation=evaluation,
        cmab_filter=cmab_filter,
        config=config,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_policy(
    model: GFlowNet,
    env: SubagentEnvironment,
    *,
    n_samples: int = 1000,
    active_arms: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> EvaluationReport:
    """Sample ``n_samples`` terminal subsets from ``P_F`` and compare to ``R``.

    A correctly-trained GFN should produce empirical frequencies roughly
    proportional to ``R(x) / Z`` *restricted to the CMAB-bounded
    subspace*. The report exposes both the empirical histogram and the
    analytic target (with the same active-arms restriction) so the CLI
    can show a clean apples-to-apples comparison.
    """
    model.eval()
    terminals, *_ = sample_trajectories(
        model, env, n_samples, active_arms=active_arms, device=device
    )
    counts: Counter[str] = Counter()
    sizes = 0
    for i in range(n_samples):
        subset = env.subset_from_state(terminals[i])
        counts[subset_to_id(subset)] += 1
        sizes += len(subset)
    subset_counts = dict(counts)
    subset_freqs = {k: v / n_samples for k, v in subset_counts.items()}

    # Analytic target: R/Z over the CMAB-bounded terminal set.
    if active_arms is None:
        active = set(TOOLS)
    else:
        active = {t for t, on in zip(TOOLS, active_arms.tolist(), strict=False) if on}
    target: dict[str, float] = {}
    z = 0.0
    for subset, util in env.utilities.items():
        if subset.issubset(active):
            r = math.exp(util / env.temperature)
            target[subset_to_id(subset)] = r
            z += r
    target_freqs = {k: v / z for k, v in target.items()} if z > 0 else {}

    top1_share = max(subset_freqs.values()) if subset_freqs else 0.0
    return EvaluationReport(
        n_samples=n_samples,
        subset_counts=subset_counts,
        subset_freqs=subset_freqs,
        target_freqs=target_freqs,
        mode_share_top1=top1_share,
        unique_terminals=len(subset_counts),
        avg_subset_size=sizes / n_samples,
        learned_log_z=float(model.log_z.item()),
    )
