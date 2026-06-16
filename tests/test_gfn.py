"""Tests for the CMAB-GFN explorer (Phase 1).

The GFN suite needs ``torch``, which lives behind the ``[gfn]`` extra so
that the default install of ``gpqa-cmab`` stays slim. Whenever ``torch``
is not importable we skip the entire module instead of failing the
default test run.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from gpqa_cmab.gfn import (  # noqa: E402  (after importorskip)
    EMPIRICAL_UTILITIES,
    TOOLS,
    CMABFilter,
    SubagentEnvironment,
    marginal_contributions,
    single_arm_utilities,
    subset_to_id,
)
from gpqa_cmab.gfn.model import GFlowNet  # noqa: E402
from gpqa_cmab.gfn.training import (  # noqa: E402
    sample_trajectories,
    train_cmab_gfn,
    trajectory_balance_loss,
)


# --- empirical data ----------------------------------------------------------
def test_empirical_table_has_all_16_subsets():
    assert len(EMPIRICAL_UTILITIES) == 16
    # All utilities strictly positive — required for log R(x) to stay finite.
    assert all(v > 0.0 for v in EMPIRICAL_UTILITIES.values())


def test_subset_to_id_canonical_ordering():
    assert subset_to_id({"C", "A"}) == "A,C"
    assert subset_to_id(set()) == "main_only"
    assert subset_to_id({"A", "B", "C", "D"}) == "A,B,C,D"


# --- CMAB filter -------------------------------------------------------------
def test_single_arm_utility_filter_drops_low_arms():
    flt = CMABFilter.from_single_arm_utility(gamma=0.6)
    active = [t for t, on in zip(TOOLS, flt.active_arms, strict=True) if on]
    # On the MVP empirical data A=0.743, B=0.542, C=0.797, D=0.568.
    # gamma=0.6 keeps A and C only.
    assert active == ["A", "C"]
    assert flt.mask.dtype == torch.bool


def test_marginal_contribution_filter_signs_match_intuition():
    scores = marginal_contributions()
    # On the MVP data the strongest contributor is C; the weakest is D.
    assert scores["C"] == max(scores.values())
    assert scores["D"] == min(scores.values())


def test_solo_utilities_match_table():
    solo = single_arm_utilities()
    assert solo["A"] == pytest.approx(0.743)
    assert solo["C"] == pytest.approx(0.797)


def test_all_active_filter_is_no_op():
    flt = CMABFilter.all_active()
    assert all(flt.active_arms)


# --- environment -------------------------------------------------------------
def test_action_mask_excludes_already_picked_and_pruned_arms():
    env = SubagentEnvironment()
    state = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # {A}
    active = torch.tensor([True, False, True, False])
    mask = env.action_mask(state, active)
    # ADD_A invalid (already in), ADD_B invalid (pruned), ADD_C valid,
    # ADD_D invalid (pruned), TERMINATE always valid.
    assert mask.tolist() == [[False, False, True, False, True]]


def test_action_mask_terminate_always_allowed():
    env = SubagentEnvironment()
    state = torch.zeros(1, 4)
    mask = env.action_mask(state, active_arms=None)
    assert mask[0, env.terminate_action].item() is True


def test_reward_is_strictly_positive_and_monotonic_in_utility():
    env = SubagentEnvironment(temperature=0.1)
    states = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],  # main_only u=0.436
            [1.0, 0.0, 1.0, 0.0],  # A,C       u=0.801
            [0.0, 1.0, 0.0, 0.0],  # B         u=0.542
        ]
    )
    rewards = env.reward(states)
    assert (rewards > 0).all().item()
    # A,C has the highest utility -> highest reward.
    assert rewards[1] > rewards[0]
    assert rewards[1] > rewards[2]


# --- model -------------------------------------------------------------------
def test_gflownet_log_policy_respects_action_mask():
    model = GFlowNet(n_tools=4, n_actions=5, hidden_dim=8)
    state = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    mask = torch.tensor([[False, True, False, True, True]])
    log_p = model.log_policy(state, mask)
    probs = log_p.exp()
    # Masked actions have ~zero probability.
    assert probs[0, 0].item() < 1e-6
    assert probs[0, 2].item() < 1e-6
    # Allowed actions sum (almost exactly) to 1.
    assert math.isclose(
        probs[0, [1, 3, 4]].sum().item(), 1.0, rel_tol=1e-5, abs_tol=1e-5
    )


def test_gflownet_has_learnable_log_z():
    model = GFlowNet()
    assert model.log_z.requires_grad
    assert model.log_z.shape == torch.Size([1])


# --- training loop -----------------------------------------------------------
def test_sample_trajectories_only_emits_active_arms():
    torch.manual_seed(0)
    env = SubagentEnvironment(temperature=0.1)
    model = GFlowNet(n_tools=env.n_tools, n_actions=env.n_actions, hidden_dim=8)
    active = torch.tensor([True, False, True, False])  # only A and C
    terminals, log_pf, log_pb, log_r = sample_trajectories(
        model, env, batch_size=32, active_arms=active
    )
    # No trajectory should contain B (idx 1) or D (idx 3).
    forbidden = terminals[:, [1, 3]]
    assert torch.all(forbidden == 0).item()
    # Shapes line up.
    assert log_pf.shape == (32,)
    assert log_pb.shape == (32,)
    assert log_r.shape == (32,)


def test_trajectory_balance_loss_is_non_negative():
    torch.manual_seed(0)
    env = SubagentEnvironment()
    model = GFlowNet(n_tools=env.n_tools, n_actions=env.n_actions, hidden_dim=8)
    loss = trajectory_balance_loss(model, env, batch_size=16)
    assert loss.item() >= 0.0


def test_short_training_run_decreases_tb_loss():
    """End-to-end: a small training run should drive TB loss well below
    the initial value and yield an empirical sampling distribution that
    concentrates mass on the high-utility subsets inside the CMAB-bounded
    subspace."""
    torch.manual_seed(0)
    env = SubagentEnvironment(temperature=0.1)
    flt = CMABFilter.from_single_arm_utility(gamma=0.6)
    result = train_cmab_gfn(
        env=env,
        cmab_filter=flt,
        num_iters=600,
        batch_size=32,
        seed=0,
        eval_samples=500,
        checkpoint_every=200,
    )
    # Loss should have meaningfully decreased.
    assert result.history.losses[-1] < result.history.losses[0]
    # CMAB filter keeps A, C only -> 4 reachable terminal subsets.
    expected_terminals = {"main_only", "A", "C", "A,C"}
    assert set(result.evaluation.subset_freqs).issubset(expected_terminals)
    # The single dominant mode must NOT take more than ~80% of mass -
    # we want diversity-preserving sampling, not collapse.
    assert result.evaluation.mode_share_top1 < 0.8
    # Best subset A,C (or solo C) should be in the top two.
    top2 = sorted(result.evaluation.subset_freqs.items(), key=lambda kv: -kv[1])[:2]
    top2_ids = {k for k, _ in top2}
    assert "A,C" in top2_ids or "C" in top2_ids
