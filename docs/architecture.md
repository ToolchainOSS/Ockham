# Architecture Overview

This MVP is a local Python research pipeline. There is no web frontend, no
background service, and no persistent database. Each command reads input files,
writes JSONL/JSON/CSV/Markdown artifacts under `artifacts/`, and exits.

## Goals and non-goals

**Goals**

- Measure whether a cost-aware Combinatorial Multi-Armed Bandit (CMAB) can prune
  optional subagents while preserving most of the all-four pipeline's
  capability.
- Provide clean cost accounting (token usage per call, per subset, per
  question).
- Be fully reproducible offline through deterministic mock mode.

**Non-goals**

- Maximizing GPQA-Diamond leaderboard accuracy.
- Implementing a deployment system, web UI, or remote storage.
- Supporting arbitrary tool-use or code execution by subagents.

## Module map

```text
src/gpqa_cmab/
├── cli.py                 Typer-free argparse CLI; one function per command.
├── config.py              Env-driven Settings dataclass with @cache.
├── dataset.py             JSONL/CSV loader, normalizer, domain filter.
├── subsets.py             Deterministic 16-subset enumeration + subset IDs.
├── schemas.py             Pydantic models for LLM I/O, telemetry, results.
├── prompts.py             Versioned prompt loader (prompts/ folder).
├── json_utils.py          JSON retry + validated-LLM-call helper.
├── telemetry.py           Per-call and per-subset aggregation, JSONL writer.
├── metrics.py             Accuracy, utility, bootstrap, baselines.
├── reporting.py           Markdown report + CSV/JSON evaluation outputs.
├── llm/
│   ├── base.py            LLMClient ABC.
│   ├── mock.py            Deterministic mock provider.
│   └── openai_client.py   Real OpenAI adapter (optional install).
├── agents/
│   ├── subagents.py       A/B/C/D runners (independent, parallelizable).
│   ├── main_integrator.py Consumes selected subagent reports.
│   └── self_consistency.py CoT-K plurality-vote baseline.
├── bandits/
│   ├── superarm_ts.py     Beta-Bernoulli super-arm Thompson sampling.
│   └── structured_cmab.py Shared-feature logistic CMAB with TS-style bonus.
└── experiments/
    ├── factorial.py       Full 16-subset factorial sweep.
    ├── replay.py          Partial-information bandit replay.
    └── self_consistency.py CoT-1/SC-K dataset runner.
```

## Runtime data flow

```text
                ┌───────────────────────────────────────────────┐
                │ dataset.py: load_questions(input, domain)      │
                └───────────────────────────────────────────────┘
                                  │ List[GPQAQuestion]
                                  ▼
        ┌──────────────────────────────────────────────────────────┐
        │ agents/subagents.run_all_subagents (A,B,C,D parallel)     │
        │  └─ uses json_utils.complete_validated → telemetry.record │
        └──────────────────────────────────────────────────────────┘
                                  │ dict[agent → SubagentReport]
                                  ▼
        ┌──────────────────────────────────────────────────────────┐
        │ For each S in subsets.all_subsets() (16 total):           │
        │   agents/main_integrator.run_main_integrator(S-reports)   │
        │   → telemetry.aggregate_usage  → FactorialResult          │
        └──────────────────────────────────────────────────────────┘
                                  │ List[FactorialResult]
                                  ▼
        ┌──────────────────────────────────────────────────────────┐
        │ metrics + reporting.write_evaluation_outputs              │
        │   metrics_summary.json, subset_accuracy_table.csv         │
        └──────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────────────┐
        │ experiments/replay.replay_bandit (partial information):    │
        │   superarm-ts or structured-cmab                          │
        │   per step observes ONLY selected subset row              │
        └──────────────────────────────────────────────────────────┘
                                  │ List[BanditStep]
                                  ▼
        ┌──────────────────────────────────────────────────────────┐
        │ reporting.write_report → artifacts/reports/mvp_report.md  │
        └──────────────────────────────────────────────────────────┘
```

## Module boundaries (invariants)

- **Dataset loading is separate from agent execution.** `dataset.py` never
  imports `llm/`, `agents/`, or `experiments/`.
- **LLM I/O always goes through `LLMClient.complete()`**, which is wrapped by
  `json_utils.complete_validated()` so that telemetry, retries, and error
  logging are guaranteed.
- **Telemetry never lives inside the bandit code.** Bandits consume
  `FactorialResult` rows; they do not call LLMs.
- **Bandit replay must obey partial information.** See
  [cmab.md](cmab.md#partial-information-replay-protocol).
- **Reports never read raw question text** unless explicitly configured to do
  so. They use question IDs and aggregate counts.

## Configuration surface

Configuration lives in `gpqa_cmab.config.Settings`, populated from environment
variables. See [.env.example](../.env.example) for the full list and
[development.md](development.md#environment-and-secrets) for guidance on
secrets.

## Concurrency model

The MVP runs single-process and single-threaded. Subagents A/B/C/D are
*independent and parallelizable*, but the reference implementation invokes them
sequentially. The independence claim is structural: no subagent depends on
another's output. Parallelism can be added later by replacing the for-loop in
`agents/subagents.run_all_subagents` with `concurrent.futures`.
