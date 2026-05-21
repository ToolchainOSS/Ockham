# CLI Reference

All commands are exposed under the `gpqa-cmab` entry point. Invoke them through
`uv run gpqa-cmab <command> …` so that the project's locked dependencies are
used.

Every command exits non-zero on error and emits a single JSON status line on
stdout. Detailed logs go through the standard `logging` module; configure
verbosity via `LOG_LEVEL`.

## Common flags

| Flag | Where | Purpose |
|---|---|---|
| `--input PATH` | data-consuming commands | JSONL or CSV file with GPQA records. |
| `--domain NAME` | data-consuming commands | Domain filter, default `physics`. |
| `--max-questions N` | data-consuming commands | Cap dataset size for dry runs and cost control. |
| `--output PATH` | producers | Output artifact path. |

## `validate-data`

Sanity-check a dataset file. Loads, normalizes, and counts rows after the
domain filter.

```bash
uv run gpqa-cmab validate-data --input data/gpqa_diamond.csv --domain physics
```

## `run-subagents`

Run A/B/C/D once per question and cache the JSON reports plus telemetry.

```bash
uv run gpqa-cmab run-subagents \
  --input data/gpqa_diamond.csv \
  --domain physics \
  --output artifacts/cache/subagent_cache.jsonl
```

## `run-factorial`

Run the full 16-subset factorial sweep. The runner calls each subagent once per
question (4 calls) and the main integrator 16 times per question (one per
subset).

```bash
uv run gpqa-cmab run-factorial \
  --input data/gpqa_diamond.csv \
  --domain physics \
  --output artifacts/results/full_factorial_results.jsonl \
  --max-questions 20 \
  --max-api-calls 1000
```

Flags:

- `--max-api-calls N` — hard cap; the runner stops cleanly and returns the
  partial result list.
- `--dry-run` — skip LLM calls, only report the planned call count.

## `evaluate`

Compute aggregate metrics from factorial results.

```bash
uv run gpqa-cmab evaluate \
  --results artifacts/results/full_factorial_results.jsonl \
  --output-dir artifacts/results
```

Produces `subset_accuracy_table.csv` and `metrics_summary.json` (which now
embeds the static / random / oracle baseline summary).

## `replay-bandit`

Simulate a CMAB policy under the partial-information protocol over the
factorial table.

```bash
uv run gpqa-cmab replay-bandit \
  --results artifacts/results/full_factorial_results.jsonl \
  --policy superarm-ts \
  --seeds 100 \
  --output artifacts/results/bandit_replay_results.jsonl
```

Policies: `superarm-ts`, `structured-cmab`. See [cmab.md](cmab.md).

## `run-self-consistency`

Run the CoT-K plurality-vote baseline across the dataset.

```bash
uv run gpqa-cmab run-self-consistency \
  --input data/gpqa_diamond.csv \
  --domain physics \
  --output artifacts/results/self_consistency_results.jsonl \
  --k-values 1,4,8,16 \
  --seed 0 \
  --temperature 0.7
```

Each output row records K, the final voted answer, correctness, total tokens,
and call count.

## `baselines`

Compute static-pruning, random budget-matched, and oracle fixed-subset
references in one shot from the factorial table.

```bash
uv run gpqa-cmab baselines \
  --results artifacts/results/full_factorial_results.jsonl \
  --output artifacts/results/baselines_summary.json \
  --static-subset A \
  --target-subset A,B,C,D \
  --seeds 100
```

See [baselines.md](baselines.md).

## `report`

Render the human-readable Markdown report from artifacts.

```bash
uv run gpqa-cmab report \
  --results-dir artifacts/results \
  --output artifacts/reports/mvp_report.md
```

The report parses `metrics_summary.json`, `bandit_replay_results.jsonl`, and
the embedded baselines block; it computes per-policy unique subset coverage and
regret versus the oracle fixed-subset reference.

## `smoke-test`

End-to-end mock run; requires no API keys. Used by the quality gate.

```bash
uv run gpqa-cmab smoke-test --mock
```

## `quick-check`

Pick a random physics question and run it through the subagent + main
integrator pipeline. Designed to catch pipeline regressions fast and cheap:
defaults to the mock provider (zero cost, no network) and the smallest
meaningful run is just 2 LLM calls.

```bash
# Cheapest sanity check: 1 subagent + 1 main integrator call, mock provider.
uv run gpqa-cmab quick-check --subset A

# Full A/B/C/D pipeline on a random physics question (5 calls, still free).
uv run gpqa-cmab quick-check

# Reproducible pick via seed, or target a specific record id.
uv run gpqa-cmab quick-check --seed 42
uv run gpqa-cmab quick-check --question-id recoiTJPGUmzAkief
```

The command prints a single JSON object with the predicted answer, correctness,
token usage, estimated cost (using `COST_USD_PER_1K_TOKENS`), and latency.
Non-mock providers are refused unless `--allow-real-llm` is set; without that
flag the command transparently downgrades to mock so it can never burn billable
tokens by accident.
