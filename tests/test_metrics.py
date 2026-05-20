from gpqa_cmab.dataset import load_questions
from gpqa_cmab.experiments.factorial import run_full_factorial
from gpqa_cmab.llm.mock import MockLLMClient
from gpqa_cmab.metrics import bootstrap_ci, subset_table


def test_metrics_compute_accuracy_and_token_savings(sample_jsonl):
    rows = run_full_factorial(
        load_questions(sample_jsonl, "physics"),
        MockLLMClient(),
        main_model="m",
        subagent_model="s",
    )
    table = subset_table(rows)
    assert len(table) == 16
    main = next(row for row in table if row["subset_id"] == "main_only")
    assert main["accuracy"] == 1.0
    assert bootstrap_ci([1.0, 0.0], samples=20)
