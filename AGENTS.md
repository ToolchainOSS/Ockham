# AGENTS.md

Local Python research MVP: GPQA-Diamond Physics experiments on cost-aware CMAB
pruning of optional LLM subagents. The claim under test is **token and cost
reduction while preserving most of the all-four-subagent capability** — not
GPQA leaderboard accuracy.

## Agent Operating Contract

Every coding session in this repository runs under this contract. It is
vendor-neutral: it applies to any agent or tool driving changes here.

- **Autonomy:** Proceed without asking permission for routine steps. On
  ambiguity, make the most reasonable technical assumption, state it, and
  continue. Pause only for genuinely irreversible or destructive actions
  (see Boundaries).
- **State tracking:** Maintain a running TODO list for any multi-step task and
  keep it current as items move pending → active → done.
- **Delegation:** When a subtask risks overwhelming the working context (broad
  search, deep refactor), delegate it to a sub-agent / exploration routine
  rather than inlining it.
- **Termination:** Never stop silently. When the objective is verifiably
  complete and the Quality Gate passes, announce completion explicitly and
  await the next directive.

Per-step reporting loop: **Current state** (what just finished) → **Assumptions**
(independent decisions) → **TODO update** → **Next action** (the command or
tool you are about to run).

## Tooling & Commands

- **Package manager:** `uv` only. Never call `pip`, `python -m venv`, or a
  system Python directly.
- **Run the CLI:** `uv run gpqa-cmab <command> …` — see [CLI reference](docs/cli.md).
- **Run one test:** `uv run pytest tests/test_bandits.py -k <name>`.
- **Quality Gate** — all four must pass before finishing substantive changes:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest --cov=gpqa_cmab --cov-report=term-missing   # CI enforces --cov-fail-under=80
uv run gpqa-cmab smoke-test --mock
```

## Boundaries & Constraints

Each prohibition is paired with the supported path forward.

- **Scope:** Keep it Python-only. Do not add a web app, frontend, Docker
  deployment, auth, or background services — this is a local, reproducible
  research MVP.
- **LLM calls:** Never instantiate a provider SDK or HTTP client at a call
  site. Route every call through the `LLMClient` boundary and wrap structured
  calls in `complete_validated(...)` so retries, JSON validation, and telemetry
  are recorded. See [LLM boundary & telemetry](docs/telemetry.md).
- **Secrets & data:** Never commit secrets or benchmark data. Generated outputs
  go under `artifacts/` (git-ignored except `.gitkeep`).
- **Tests:** No network access in tests. Use the mock provider; mock mode must
  work without API keys. Seed every RNG for determinism.
- **Prompts & schemas:** Prompts are versioned text files in `prompts/`; outputs
  are validated Pydantic schemas. Edit the prompt file and bump its version
  instead of hard-coding prompt text in code. See [prompts & schemas](docs/prompts.md).
- **Module size:** Prefer cohesive modules under ~500 lines. When a file grows
  past that, split it into focused submodules instead of appending.
- **Errors:** Never swallow exceptions. Surface failures via explicit return
  types or raised exceptions, and record `success=False` telemetry where the
  boundary supports it.

## Research Invariants

- Subagents A, B, C, and D each receive the **same raw question plus four
  choices**. No scratch summaries, no cross-subagent communication.
- Full-factorial evaluation may compute all 16 subsets for measurement, but
  **bandit replay must only observe the selected subset's outcome** at each
  step (partial information).
- Make invalid states unrepresentable: model structured data as Pydantic
  schemas / typed records, not loose dicts.

## Code Primitive — the LLM boundary

```python
from gpqa_cmab.json_utils import complete_validated

# `client` is an LLMClient (mock or OpenAI-compatible); `request` is built
# from a versioned prompt; the third arg is the Pydantic schema the model
# output must satisfy. Retries, JSON repair, and telemetry are handled here.
report, call_telemetry = complete_validated(
    client,
    request,
    SubagentAReport,
    telemetry=telemetry,
    record_kwargs=record_kwargs,
)
```

## Domain Documentation (load on demand)

Read the file matching your task instead of pre-loading everything.

| Task | Read |
|---|---|
| Module map / data flow | [architecture.md](docs/architecture.md) |
| CLI commands & flags | [cli.md](docs/cli.md) |
| Dataset format & filtering | [dataset.md](docs/dataset.md) |
| Prompts & JSON schemas | [prompts.md](docs/prompts.md) |
| LLM boundary, retries, telemetry | [telemetry.md](docs/telemetry.md) |
| Durable event store (SQLite/Postgres) | [telemetry_db.md](docs/telemetry_db.md) |
| Provider configuration (vendor-neutral) | [providers.md](docs/providers.md) |
| CMAB / Thompson sampling design | [cmab.md](docs/cmab.md) |
| GFlowNet explorer | [gfn.md](docs/gfn.md) |
| Baselines & metrics | [baselines.md](docs/baselines.md) |
| Experiment recipes | [experiments.md](docs/experiments.md) |
| Live-run / cost runbook | [runbook.md](docs/runbook.md) |
| Dev workflow & contribution rules | [development.md](docs/development.md) |
| Design decisions (ADRs) | [decisions/README.md](docs/decisions/README.md) |
