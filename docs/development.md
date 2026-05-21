# Development Workflow

## Prerequisites

- Python 3.11 or newer (the `.python-version` file pins the project).
- [`uv`](https://docs.astral.sh/uv/) for dependency management.

## Setup

```bash
uv sync --all-extras --dev
cp .env.example .env
```

Mock mode runs out of the box. Real provider keys live in `.env` and are
**never committed**. See [telemetry.md](telemetry.md) for provider wiring.

## Quality gate

Run before every PR and before declaring a substantive change complete:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest --cov=gpqa_cmab --cov-report=term-missing
uv run gpqa-cmab smoke-test --mock
```

These four commands match the CI pipeline in
[`.github/workflows/`](../.github/workflows). Coverage target: ≥ 80% on core
modules. Tests must not require network access.

## Coding conventions

- Python 3.11+ syntax (use `X | Y`, `dict[str, int]`, `from __future__ import
  annotations`).
- Typed dataclasses or Pydantic models for structured data.
- `pathlib.Path` instead of string path manipulation.
- `logging` instead of `print`, except for user-facing CLI output (a single
  JSON status line per command).
- Small, testable functions. No global mutable state outside of `@cache`d
  factories like `gpqa_cmab.config.get_settings`.
- No frontend dependencies. No web frameworks. No background services.

## Adding code

| Change | Where | What else to update |
|---|---|---|
| New CLI command | `cli.py` | `docs/cli.md`, `tests/test_cli_smoke.py` if it affects smoke. |
| New agent | `agents/`, `schemas.py`, `prompts/`, `llm/mock.py` | `docs/prompts.md`. |
| New bandit policy | `bandits/`, `experiments/replay.py` | `docs/cmab.md`, `tests/test_bandits.py`. |
| New baseline | `metrics.py`, `reporting.py` | `docs/baselines.md`, tests. |
| New telemetry field | `schemas.py`, `telemetry.py`, `json_utils.complete_validated` | `docs/telemetry.md`, tests. |

## Testing conventions

- Tests live under `tests/` and mirror the package layout.
- Mock provider only. Tests that would hit a real API are not allowed.
- Determinism is a hard requirement; seed every RNG.
- Use the `sample_jsonl` fixture in [`tests/conftest.py`](../tests/conftest.py)
  for question fixtures instead of hand-rolling JSON in each test.

## Environment and secrets

- `.env.example` lists every supported variable. Copy to `.env` and edit
  locally.
- Never commit secrets. The repository's `.gitignore` excludes `.env`.
- CI runs without any provider keys; the mock smoke test must always pass.

## Releasing changes

This is a research MVP, not a package on PyPI. There are no version tags or
release artifacts. Squash-merge PRs once the quality gate passes locally and in
CI.

## When AI agents collaborate

- Read [AGENTS.md](../AGENTS.md) first. It enumerates invariants the codebase
  expects.
- Use [`json_utils.complete_validated`](../src/gpqa_cmab/json_utils.py)
  whenever an agent needs to call an LLM. Do not bypass it.
- Update the relevant doc in `docs/` when the contract of a module changes.
- If you change the smoke-test output, update
  [`tests/test_cli_smoke.py`](../tests/test_cli_smoke.py) in the same PR.
- Record substantive design decisions in
  [`docs/decisions/`](decisions/README.md).
