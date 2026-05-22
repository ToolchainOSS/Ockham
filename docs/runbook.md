# Runbook: Exact Commands

This document is the operator checklist for local testing, mock validation, and a real paid run. Run every command from the repository root unless noted.

## 1. Install and Configure

Install all runtime, provider, and development dependencies:

```bash
uv sync --all-extras --dev
```

Create a local environment file:

```bash
cp .env.example .env
```

Mock mode needs no secrets. For a paid OpenAI-compatible run, edit `.env` or export these variables in the shell that will run the experiment:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY='sk-...'
export MAIN_MODEL='gpt-4o-mini'
export SUBAGENT_MODEL='gpt-4o-mini'
export SELF_CONSISTENCY_MODEL='gpt-4o-mini'
export MAX_OUTPUT_TOKENS=2048
export LLM_JSON_MAX_RETRIES=2
export MAX_TOTAL_API_CALLS=1000
export MAX_TOTAL_COST_USD=10
export COST_INPUT_USD_PER_1M_TOKENS=0.15
export COST_CACHED_INPUT_USD_PER_1M_TOKENS=0.075
export COST_OUTPUT_USD_PER_1M_TOKENS=0.60
```

Replace the pricing values with the provider's current model-specific rates. Use uncached input, cached input, and output rates separately. `COST_USD_PER_1K_TOKENS` is only a legacy blended fallback when tiered rates are unavailable.

## 2. Development Quality Gate

Run the full local quality gate before changing experiment settings or starting a paid run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest --cov=gpqa_cmab --cov-report=term-missing
uv run gpqa-cmab smoke-test --mock
```

These commands verify formatting, lint rules, deterministic tests, coverage, and an offline end-to-end pipeline. The smoke test writes mock artifacts under `artifacts/results/` and `artifacts/reports/`.

## 3. Dataset Validation

Check that the dataset loads and that the domain filter is non-empty:

```bash
uv run gpqa-cmab validate-data \
  --input data/gpqa_diamond.csv \
  --domain physics
```

Use `--max-questions N` to verify a planned subset:

```bash
uv run gpqa-cmab validate-data \
  --input data/gpqa_diamond.csv \
  --domain physics \
  --max-questions 20
```

A zero-row filtered dataset is an error for experiment commands; do not proceed until this count is correct.

## 4. Cheap Pipeline Checks

Run a single-subset mock check:

```bash
uv run gpqa-cmab quick-check \
  --input data/gpqa_diamond.csv \
  --subset A \
  --seed 0
```

Run the default full-factorial mock check on one sampled question:

```bash
uv run gpqa-cmab quick-check \
  --input data/gpqa_diamond.csv \
  --seed 0
```

For a real-provider one-question check, require explicit opt-in and tight caps:

```bash
uv run gpqa-cmab quick-check \
  --input data/gpqa_diamond.csv \
  --seed 0 \
  --allow-real-llm \
  --max-api-calls 20 \
  --max-estimated-cost-usd 1
```

The JSON output includes `trace`, `log`, `manifest`, `api_calls`, token totals, `cost_breakdown`, and `estimated_cost_usd`.

## 5. Optional Subagent Cache

Build a reusable A/B/C/D cache once per question. This costs four subagent calls per question and avoids rerunning subagents during factorial sweeps:

```bash
uv run gpqa-cmab run-subagents \
  --input data/gpqa_diamond.csv \
  --domain physics \
  --max-questions 20 \
  --output artifacts/cache/subagent_cache.jsonl \
  --max-api-calls 80 \
  --max-estimated-cost-usd 2
```

Outputs:

- `artifacts/cache/subagent_cache.jsonl`
- `artifacts/cache/subagent_cache.jsonl.trace.jsonl`
- `artifacts/cache/subagent_cache.jsonl.log`
- `artifacts/cache/subagent_cache.jsonl.manifest.json`

## 6. Full Factorial Measurement

Dry-run first to confirm planned main-integrator calls:

```bash
uv run gpqa-cmab run-factorial \
  --input data/gpqa_diamond.csv \
  --domain physics \
  --max-questions 20 \
  --output artifacts/results/full_factorial_results.jsonl \
  --dry-run
```

Run factorial without a subagent cache. This is 20 calls per question: four subagents plus 16 main-integrator subsets.

```bash
uv run gpqa-cmab run-factorial \
  --input data/gpqa_diamond.csv \
  --domain physics \
  --max-questions 20 \
  --output artifacts/results/full_factorial_results.jsonl \
  --max-api-calls 400 \
  --max-estimated-cost-usd 10
```

Run factorial with a cache. This is 16 new calls per question; cached subagent telemetry is copied into the active trace for audit but not charged as new spend.

```bash
uv run gpqa-cmab run-factorial \
  --input data/gpqa_diamond.csv \
  --domain physics \
  --max-questions 20 \
  --subagent-cache artifacts/cache/subagent_cache.jsonl \
  --output artifacts/results/full_factorial_results.jsonl \
  --max-api-calls 320 \
  --max-estimated-cost-usd 8
```

Primary outputs:

- `artifacts/results/full_factorial_results.jsonl`
- `artifacts/results/full_factorial_results.jsonl.trace.jsonl`
- `artifacts/results/full_factorial_results.jsonl.log`
- `artifacts/results/full_factorial_results.jsonl.manifest.json`

## 7. Evaluate Factorial Results

Compute subset metrics and summary files:

```bash
uv run gpqa-cmab evaluate \
  --results artifacts/results/full_factorial_results.jsonl \
  --output-dir artifacts/results
```

Outputs:

- `artifacts/results/subset_accuracy_table.csv`
- `artifacts/results/metrics_summary.json`
- `artifacts/results/evaluate_manifest.json`

## 8. Replay Bandit Policies

Replay Super-arm Thompson Sampling:

```bash
uv run gpqa-cmab replay-bandit \
  --results artifacts/results/full_factorial_results.jsonl \
  --policy superarm-ts \
  --seeds 100 \
  --output artifacts/results/bandit_replay_results.jsonl
```

Replay Structured CMAB:

```bash
uv run gpqa-cmab replay-bandit \
  --results artifacts/results/full_factorial_results.jsonl \
  --policy structured-cmab \
  --seeds 100 \
  --output artifacts/results/structured_cmab_replay_results.jsonl
```

Bandit replay is offline and must observe only the selected subset outcome at each step.

## 9. Self-Consistency Baseline

Run CoT-K plurality-vote baselines with strict caps:

```bash
uv run gpqa-cmab run-self-consistency \
  --input data/gpqa_diamond.csv \
  --domain physics \
  --max-questions 20 \
  --k-values 1,4,8,16 \
  --seed 0 \
  --temperature 0.7 \
  --output artifacts/results/self_consistency_results.jsonl \
  --max-api-calls 580 \
  --max-estimated-cost-usd 10
```

The planned call count is `max_questions * sum(k_values)`. For the command above, that is `20 * 29 = 580` calls.

## 10. Static, Random, and Oracle Baselines

Compute non-LLM baselines from the factorial table:

```bash
uv run gpqa-cmab baselines \
  --results artifacts/results/full_factorial_results.jsonl \
  --output artifacts/results/baselines_summary.json \
  --static-subset A \
  --target-subset A,B,C,D \
  --seeds 100
```

## 11. Report Generation

Render the Markdown report:

```bash
uv run gpqa-cmab report \
  --results-dir artifacts/results \
  --output artifacts/reports/mvp_report.md
```

The report reads `metrics_summary.json`, bandit replay artifacts, self-consistency results when present, and baseline summaries.

## 12. First Paid Production Run Checklist

Run these in order:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest --cov=gpqa_cmab --cov-report=term-missing
uv run gpqa-cmab smoke-test --mock
uv run gpqa-cmab validate-data --input data/gpqa_diamond.csv --domain physics
uv run gpqa-cmab quick-check --input data/gpqa_diamond.csv --seed 0
uv run gpqa-cmab quick-check --input data/gpqa_diamond.csv --seed 0 --allow-real-llm --max-api-calls 20 --max-estimated-cost-usd 1
uv run gpqa-cmab run-factorial --input data/gpqa_diamond.csv --domain physics --max-questions 20 --output artifacts/results/full_factorial_results.jsonl --dry-run
uv run gpqa-cmab run-factorial --input data/gpqa_diamond.csv --domain physics --max-questions 20 --output artifacts/results/full_factorial_results.jsonl --max-api-calls 400 --max-estimated-cost-usd 10
uv run gpqa-cmab evaluate --results artifacts/results/full_factorial_results.jsonl --output-dir artifacts/results
uv run gpqa-cmab replay-bandit --results artifacts/results/full_factorial_results.jsonl --policy superarm-ts --seeds 100 --output artifacts/results/bandit_replay_results.jsonl
uv run gpqa-cmab run-self-consistency --input data/gpqa_diamond.csv --domain physics --max-questions 20 --k-values 1,4,8,16 --seed 0 --temperature 0.7 --output artifacts/results/self_consistency_results.jsonl --max-api-calls 580 --max-estimated-cost-usd 10
uv run gpqa-cmab baselines --results artifacts/results/full_factorial_results.jsonl --output artifacts/results/baselines_summary.json --static-subset A --target-subset A,B,C,D --seeds 100
uv run gpqa-cmab report --results-dir artifacts/results --output artifacts/reports/mvp_report.md
```

Archive all result JSONL files, trace JSONL files, log files, manifest files, generated summaries, generated reports, the `prompts/` directory, and the git commit recorded in each manifest. The manifest checksums are the integrity record for paper submission.
