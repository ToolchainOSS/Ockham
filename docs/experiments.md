# Running Experiments

## Mock end-to-end

The mock smoke test is the canonical "does everything still work" recipe:

```bash
uv run gpqa-cmab smoke-test --mock
```

It generates a one-question dataset, runs the factorial, evaluates metrics,
replays the super-arm TS bandit for two seeds, runs a small self-consistency
sweep (K=1, K=4), and renders `artifacts/reports/mvp_report.md`.

## Small live experiment

Cost-cap first, scale later. With a small dataset:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=...        # or AZURE_OPENAI_* if using Azure
export MAIN_MODEL=gpt-4o-mini
export SUBAGENT_MODEL=gpt-4o-mini
export COST_INPUT_USD_PER_1M_TOKENS=0.15
export COST_CACHED_INPUT_USD_PER_1M_TOKENS=0.075
export COST_OUTPUT_USD_PER_1M_TOKENS=0.60
export MAX_TOTAL_COST_USD=10

uv run gpqa-cmab validate-data --input data/gpqa_diamond.csv --domain physics

uv run gpqa-cmab run-factorial \
  --input data/gpqa_diamond.csv \
  --domain physics \
  --output artifacts/results/full_factorial_results.jsonl \
  --max-questions 20 \
  --max-api-calls 1000

uv run gpqa-cmab evaluate \
  --results artifacts/results/full_factorial_results.jsonl \
  --output-dir artifacts/results

uv run gpqa-cmab replay-bandit \
  --results artifacts/results/full_factorial_results.jsonl \
  --policy structured-cmab \
  --seeds 100 \
  --output artifacts/results/bandit_replay_results.jsonl

uv run gpqa-cmab run-self-consistency \
  --input data/gpqa_diamond.csv \
  --domain physics \
  --k-values 1,4,8 \
  --max-questions 20 \
  --output artifacts/results/self_consistency_results.jsonl

uv run gpqa-cmab report \
  --results-dir artifacts/results \
  --output artifacts/reports/mvp_report.md
```

## Reproducibility checklist

- **Prompt versions** are recorded in every `FactorialResult.prompt_versions`.
- **Model names** come from `MAIN_MODEL` / `SUBAGENT_MODEL` /
  `SELF_CONSISTENCY_MODEL` and appear in every `CallTelemetry` row.
- **Random seeds** must be set explicitly (`--seeds`, `--seed`) and are
  echoed into bandit and self-consistency rows.
- **Subset IDs** are deterministic; the test suite includes a stability
  check.
- **Experiment IDs** are auto-generated with `uuid4` and recorded on every
  row; pass `experiment_id` explicitly when scripting comparisons.

## Statistical framing

The MVP sample size is small. Report:

- Raw accuracy and cost-aware utility with bootstrap CIs (`bootstrap_ci`).
- McNemar-style discordance counts for paired comparisons.
- A non-inferiority assessment: `accuracy_gap = acc_all_four - acc_cmab`;
  declare non-inferior when `accuracy_gap <= epsilon` (default
  `epsilon=0.03`).

Avoid claiming "significance" without bootstrap support. The brief explicitly
warns against overclaiming.

## Cost guardrails

Every command that issues API calls supports `--max-questions`; `run-factorial`
additionally supports `--max-api-calls` and `--dry-run`. Use them. For very
expensive sweeps, run `--dry-run` first to confirm the planned call count.

For exact command sequences from setup through paper artifact archival, use the
[runbook](runbook.md).
