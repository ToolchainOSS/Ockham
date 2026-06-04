import math

from gpqa_cmab.bandits.structured_cmab import INTERCEPT_IDX, StructuredCMAB
from gpqa_cmab.bandits.superarm_ts import SuperArmThompsonSampler
from gpqa_cmab.dataset import load_questions
from gpqa_cmab.experiments.factorial import run_full_factorial
from gpqa_cmab.experiments.replay import replay_bandit
from gpqa_cmab.llm.mock import MockLLMClient


def test_superarm_updates_posterior():
    # Default (post-fix) prior is Beta(3, 2). +1 win for A / +1 loss for B.
    bandit = SuperArmThompsonSampler(seed=1)
    bandit.update("A", True)
    bandit.update("B", False)
    assert bandit.success["A"] == 4.0
    assert bandit.failure["B"] == 3.0


def test_superarm_legacy_uniform_prior_still_available():
    # Set the explicit legacy prior for ablation / repro of older runs.
    bandit = SuperArmThompsonSampler(seed=1, alpha0=1.0, beta0=1.0)
    bandit.update("A", True)
    assert bandit.success["A"] == 2.0


def test_structured_cmab_shared_features_update_weights():
    bandit = StructuredCMAB(seed=1)
    before = bandit.weights[:]
    bandit.update("A,C,D", True, token_cost=10, avg_all_four_tokens=20)
    assert bandit.weights != before


def test_structured_cmab_warm_starts_intercept():
    # The cold-start collapse bug was rooted in a 0-initialised intercept,
    # which forced σ(0)=0.5 on step 1 and let the cost penalty pick
    # ``main_only`` every time. The fix warm-starts intercept to logit(0.7).
    bandit = StructuredCMAB(seed=0)
    expected = math.log(0.7 / 0.3)
    assert math.isclose(bandit.weights[INTERCEPT_IDX], expected, rel_tol=1e-6)


def test_structured_cmab_legacy_mode_recreates_collapse():
    # Recovering the buggy legacy behaviour must still be possible for
    # ablation/repro: zero-intercept + L2-shrinks-intercept + inv-sqrt bonus.
    bandit = StructuredCMAB(
        seed=0,
        prior_accuracy=0.5,
        shrink_intercept=True,
        uncertainty=0.1,
        bonus_form="inv_sqrt_n",
    )
    assert bandit.weights[INTERCEPT_IDX] == 0.0


def test_structured_cmab_does_not_l2_shrink_intercept():
    # With L2 OFF for the intercept, the intercept update reduces to the
    # plain gradient ``lr * (error - 0)``. Compare against the legacy
    # behaviour where the L2 term also subtracted ``lr * l2 * weight``.
    bandit = StructuredCMAB(seed=0)
    bandit.weights[INTERCEPT_IDX] = 5.0
    legacy = StructuredCMAB(seed=0, shrink_intercept=True)
    legacy.weights[INTERCEPT_IDX] = 5.0
    bandit.update("main_only", True, token_cost=900, avg_all_four_tokens=8400)
    legacy.update("main_only", True, token_cost=900, avg_all_four_tokens=8400)
    # Both share the same data gradient; the difference is exactly the
    # L2 penalty term: lr * l2 * intercept = 0.2 * 0.01 * 5.0 = 0.01.
    delta = bandit.weights[INTERCEPT_IDX] - legacy.weights[INTERCEPT_IDX]
    assert math.isclose(delta, 0.2 * 0.01 * 5.0, rel_tol=1e-6, abs_tol=1e-9)


def test_replay_only_emits_selected_subset_observations(sample_jsonl):
    rows = run_full_factorial(
        load_questions(sample_jsonl, "physics"),
        MockLLMClient(),
        main_model="m",
        subagent_model="s",
    )
    steps = replay_bandit(rows, policy="superarm-ts", seeds=3)
    assert len(steps) == 3
    assert all(step.selected_subset_id for step in steps)
    assert all(step.unique_subsets_explored == 1 for step in steps)
