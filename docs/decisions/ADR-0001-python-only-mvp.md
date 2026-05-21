# ADR-0001: Python-only local MVP (no frontend, no web service)

- Status: Accepted
- Date: 2026-05-21

## Context

The repository was scaffolded from a previous research project that included
web frontend code, Docker services, and authentication scaffolding. The
research goal is cost-aware subagent pruning measured on GPQA-Diamond Physics.
That goal does not require a UI, a backend service, or remote storage; it
requires deterministic, reproducible local experiments and clean cost
accounting.

## Decision

Keep the project Python-only. Remove all frontend, web service, and
authentication code. Generate artifacts under `artifacts/` and ignore them in
git. Use `uv` for dependency management and provide a single CLI entry point
(`gpqa-cmab`).

## Consequences

- **Positive**: small surface area, easy CI, no secrets in CI, reproducible by
  any researcher with a working `uv` environment.
- **Positive**: AGENTS.md can enforce the no-frontend invariant.
- **Negative**: no interactive dashboards. Acceptable: reports are static
  Markdown, which suffices for an MVP and integrates with PR review.

## Alternatives considered

- **Streamlit/FastAPI dashboard**: rejected — not on the research critical
  path, adds dependencies and security surface.
- **Notebook-only workflow**: rejected — hard to test in CI, encourages
  one-off state that hurts reproducibility.
