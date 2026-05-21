# Documentation Index

This directory holds design, architecture, and operational documentation for the
GPQA-Diamond Physics CMAB Subagent Pruning MVP. The top-level
[README](../README.md) is the entry point; the documents linked below go deeper
into specific aspects of the system.

## Topics

- [Architecture overview](architecture.md) — modules, data flow, runtime
  boundaries.
- [CLI reference](cli.md) — every `gpqa-cmab` subcommand with flags and
  examples.
- [Dataset contract](dataset.md) — input formats, normalization, and domain
  filtering.
- [Prompts and JSON contracts](prompts.md) — versioned prompts and the Pydantic
  schemas they must produce.
- [LLM boundary and telemetry](telemetry.md) — provider abstraction, retry
  policy, request/response logging, token accounting.
- [LLM providers](providers.md) — vendor-neutral configuration recipes
  (OpenAI, Azure, Together, Groq, OpenRouter, DeepSeek, vLLM, Ollama, …).
- [Bandits and CMAB design](cmab.md) — Super-arm Thompson Sampling, Structured
  CMAB features, reward, and partial-information replay protocol.
- [Baselines and metrics](baselines.md) — main-only, single subagent, all-four,
  static / random budget-matched pruning, oracle fixed-subset, self-consistency.
- [Running experiments](experiments.md) — end-to-end recipes for mock and live
  runs, reproducibility checklist.
- [Development workflow](development.md) — local setup, quality gate, testing
  conventions, contribution rules.

## Architecture Decision Records

ADRs capture significant design choices, the alternatives considered, and the
reasoning. They are immutable once accepted; new decisions supersede old ones.

- [ADR index](decisions/README.md)

## How to use this folder

- **Human collaborators**: start with the [architecture overview](architecture.md)
  and then jump into the topic relevant to your task.
- **AI agents and automation**: the top-level [AGENTS.md](../AGENTS.md) defines
  invariants and the quality gate. Read it first; then consult the topic doc
  closest to the file you are about to edit. Reuse helpers documented in
  [LLM boundary and telemetry](telemetry.md) instead of bypassing them.
- **Reviewers**: see [development workflow](development.md) for the local
  quality gate and PR expectations.
