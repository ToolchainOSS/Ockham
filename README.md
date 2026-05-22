# GPQA-Diamond Physics CMAB Subagent Pruning MVP

A local Python research MVP for cost-efficient LLM inference. It evaluates
whether a cost-aware Combinatorial Multi-Armed Bandit (CMAB) can prune optional
helper subagents while preserving most of the capability of an expensive
all-four-subagent pipeline on GPQA-Diamond Physics.

> This is a cost-saving inference project, not a GPQA leaderboard project.

---

## What it does

```
┌────────────────────────────────────────────────────────────────────┐
│ Question + 4 choices                                               │
│        │                                                           │
│        ▼                                                           │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────┐ │
│  │ Subagent A  │   │ Subagent B  │   │ Subagent C  │   │ Sub. D  │ │
│  │ specialist  │   │ reference   │   │ computation │   │ verifier│ │
│  └─────┬───────┘   └─────┬───────┘   └─────┬───────┘   └────┬────┘ │
│        │ (selected subset S ⊆ {A,B,C,D} only)                ▼     │
│        └──────────────────┬─────────────────────────────────►main  │
│                           ▼                                  │     │
│                ┌───────────────────────┐                     │     │
│                │ Main Integrator       │◄────────────────────┘     │
│                │ JSON: A | B | C | D   │                           │
│                └───────────┬───────────┘                           │
│                            ▼                                       │
│                   compare to gold answer                           │
│                            ▼                                       │
│                Telemetry: tokens, latency, cost                    │
└────────────────────────────────────────────────────────────────────┘
```

For each question, subagents run independently and in parallel on the same raw
input. A CMAB learns which subsets to invoke to preserve accuracy while
spending fewer tokens. See [docs/architecture.md](docs/architecture.md) for the
full data flow.

## Research claims under evaluation

| # | Claim | Metric |
|---|---|---|
| 1 | Each isolated subagent may or may not lift over main-only. | Single-subagent accuracy and utility versus `main_only`. |
| 2 | The all-four pipeline is capable but expensive. | All-four accuracy and total tokens. |
| 3 | CMAB pruning preserves capability while reducing cost. | CMAB accuracy within ~2-3 pp of all-four with ≥20% token reduction. |
| 4 | CMAB explores intelligently, not exhaustively. | Unique subsets explored, regret vs oracle fixed-subset reference. |

Details and statistical framing live in
[docs/experiments.md](docs/experiments.md) and
[docs/baselines.md](docs/baselines.md).

## Quick start

```bash
uv sync --all-extras --dev
cp .env.example .env

# Offline end-to-end check (no API keys required)
uv run gpqa-cmab smoke-test --mock
```

Real provider — works with **any OpenAI-API-compatible vendor** (OpenAI,
Azure, Together, Groq, OpenRouter, DeepSeek, vLLM, Ollama, …). See
[docs/providers.md](docs/providers.md) for the full list and recipes:

```bash
LLM_PROVIDER=openai OPENAI_API_KEY=... MAIN_MODEL=gpt-4o-mini SUBAGENT_MODEL=gpt-4o-mini \
  uv run gpqa-cmab run-factorial \
    --input data/gpqa_diamond.csv --domain physics \
    --output artifacts/results/full_factorial_results.jsonl \
    --max-questions 20
```

See [docs/cli.md](docs/cli.md) for the full command surface and
[docs/experiments.md](docs/experiments.md) for end-to-end recipes.

## Repository layout

```text
src/gpqa_cmab/      Python package — see docs/architecture.md
prompts/            Versioned LLM prompts — see docs/prompts.md
tests/              Deterministic pytest suite — see docs/development.md
artifacts/          Ignored experiment outputs (only .gitkeep is tracked)
data/               Input datasets (not committed; see docs/dataset.md)
docs/               Design, architecture, ADRs
AGENTS.md           Invariants and rules for AI collaborators
```

## Quality gate

Before every PR or substantive change:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest --cov=gpqa_cmab --cov-report=term-missing
uv run gpqa-cmab smoke-test --mock
```

Coverage target ≥ 80% on core modules. Tests must not require network access.

## Documentation

| Topic | Where |
|---|---|
| Architecture & data flow | [docs/architecture.md](docs/architecture.md) |
| CLI reference | [docs/cli.md](docs/cli.md) |
| Dataset contract | [docs/dataset.md](docs/dataset.md) |
| Prompt and JSON contracts | [docs/prompts.md](docs/prompts.md) |
| LLM boundary, retries, telemetry | [docs/telemetry.md](docs/telemetry.md) |
| LLM providers (OpenAI-compatible) | [docs/providers.md](docs/providers.md) |
| Bandit and CMAB design | [docs/cmab.md](docs/cmab.md) |
| Baselines and metrics | [docs/baselines.md](docs/baselines.md) |
| Running experiments | [docs/experiments.md](docs/experiments.md) |
| Exact command runbook | [docs/runbook.md](docs/runbook.md) |
| Development workflow | [docs/development.md](docs/development.md) |
| Architecture decision records | [docs/decisions/README.md](docs/decisions/README.md) |

## License

See [LICENSE](LICENSE).
