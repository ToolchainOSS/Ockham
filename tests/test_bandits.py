from gpqa_cmab.bandits.structured_cmab import StructuredCMAB
from gpqa_cmab.bandits.superarm_ts import SuperArmThompsonSampler
from gpqa_cmab.dataset import load_questions
from gpqa_cmab.experiments.factorial import run_full_factorial
from gpqa_cmab.experiments.replay import replay_bandit
from gpqa_cmab.llm.mock import MockLLMClient


def test_superarm_updates_posterior():
    bandit = SuperArmThompsonSampler(seed=1)
    bandit.update("A", True)
    bandit.update("B", False)
    assert bandit.success["A"] == 2
    assert bandit.failure["B"] == 2


def test_structured_cmab_shared_features_update_weights():
    bandit = StructuredCMAB(seed=1)
    before = bandit.weights[:]
    bandit.update("A,C,D", True, token_cost=10, avg_all_four_tokens=20)
    assert bandit.weights != before


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
