from gpqa_cmab.dataset import load_questions
from gpqa_cmab.experiments.factorial import run_full_factorial
from gpqa_cmab.llm.mock import MockLLMClient


def test_factorial_runner_creates_16_rows(sample_jsonl):
    questions = load_questions(sample_jsonl, "physics")
    rows = run_full_factorial(
        questions, MockLLMClient(), main_model="main", subagent_model="sub"
    )
    assert len(rows) == 16
    assert {row.subset_id for row in rows}
    assert all(row.correct for row in rows)
