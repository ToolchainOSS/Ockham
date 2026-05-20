# GPQA-Diamond Physics CMAB Subagent Pruning MVP

This is a local Python research MVP for cost-efficient LLM inference. It evaluates whether a cost-aware Combinatorial Multi-Armed Bandit can prune optional helper subagents while preserving most of the capability of an expensive all-four-subagent pipeline.

This is a cost-saving inference project, not a GPQA leaderboard project.

## Research Setup

The system has one main answer integrator and four optional subagents:

| Agent | Role |
|---|---|
| A | Physics domain-specialist solver |
| B | Reference / equation assistant |
| C | Computational / symbolic checker |
| D | Adversarial verifier / option eliminator |

Each subagent receives the same raw question plus four choices. There is no scratch summary and no cross-subagent communication. Every LLM call passes through the client boundary and telemetry recorder.

## Architecture

```text
dataset loader -> subagent runners -> main integrator -> factorial results
                                      -> metrics/reporting
                                      -> partial-information CMAB replay
```

The full factorial runner computes all 16 subsets for measurement, caching the four independent subagent reports per question. Bandit replay then simulates deployment under partial information: at each step the learner only observes the selected subset outcome.

## Project Layout

```text
src/gpqa_cmab/      Python package
prompts/            Versioned prompt contracts
tests/              Deterministic pytest suite
artifacts/          Ignored experiment outputs
```

## Setup

```bash
uv sync --all-extras --dev
cp .env.example .env
```

Mock mode requires no API keys:

```bash
uv run gpqa-cmab smoke-test --mock
```

Real OpenAI use is available through the optional adapter:

```bash
LLM_PROVIDER=openai OPENAI_API_KEY=... MAIN_MODEL=gpt-4o-mini SUBAGENT_MODEL=gpt-4o-mini \
  uv run gpqa-cmab run-factorial --input data/gpqa_diamond.csv --domain physics --output artifacts/results/full_factorial_results.jsonl
```

## Commands

Validate data:

```bash
uv run gpqa-cmab validate-data --input data/gpqa_diamond.csv --domain physics
```

Run subagents and cache reports:

```bash
uv run gpqa-cmab run-subagents \
  --input data/gpqa_diamond.csv \
  --domain physics \
  --output artifacts/cache/subagent_cache.jsonl
```

Run full factorial evaluation:

```bash
uv run gpqa-cmab run-factorial \
  --input data/gpqa_diamond.csv \
  --domain physics \
  --subagent-cache artifacts/cache/subagent_cache.jsonl \
  --output artifacts/results/full_factorial_results.jsonl \
  --max-questions 20
```

Evaluate aggregate metrics:

```bash
uv run gpqa-cmab evaluate \
  --results artifacts/results/full_factorial_results.jsonl \
  --output-dir artifacts/results
```

Replay a bandit policy:

```bash
uv run gpqa-cmab replay-bandit \
  --results artifacts/results/full_factorial_results.jsonl \
  --policy superarm-ts \
  --seeds 100 \
  --output artifacts/results/bandit_replay_results.jsonl
```

Generate the report:

```bash
uv run gpqa-cmab report \
  --results-dir artifacts/results \
  --output artifacts/reports/mvp_report.md
```

## Artifact Layout

```text
artifacts/
  cache/subagent_cache.jsonl
  results/full_factorial_results.jsonl
  results/subset_accuracy_table.csv
  results/metrics_summary.json
  results/bootstrap_results.json
  results/bandit_replay_results.jsonl
  results/bandit_summary.json
  results/self_consistency_results.jsonl
  reports/mvp_report.md
```

Generated artifacts are ignored by Git except for `artifacts/.gitkeep`.

## Metrics

The evaluator reports accuracy, average token usage, token savings versus all-four, cost per correct answer in token units, and cost-aware utility:

```text
utility = correct - lambda_token * normalized_tokens - lambda_call * num_subagents
```

Defaults are `lambda_token=0.05` and `lambda_call=0.01`. Bootstrap confidence intervals and McNemar-style paired counts are implemented in `gpqa_cmab.metrics` for analysis extensions.

## Baselines

Implemented core references include main-only, each single subagent, all-four, exhaustive oracle fixed-subset analysis, random/static pruning extension points, Super-arm Thompson Sampling, structured CMAB, and a self-consistency runner for CoT-1 and SC-K analysis.

## Limitations

Mock mode validates orchestration, schemas, telemetry, and replay behavior, but it cannot support scientific claims. A real experiment should use cost caps, multiple random seeds, enough Physics questions for uncertainty estimates, and careful reporting that avoids leaking question text.