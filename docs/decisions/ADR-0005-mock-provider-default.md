# ADR-0005: Mock provider is the default; embeds gold answer for determinism

- Status: Accepted
- Date: 2026-05-21

## Context

CI must run without API keys, and contributors must be able to validate
orchestration changes offline. We therefore need a provider that produces
schema-correct, deterministic outputs for every agent type — including
`SubagentA..D`, the main integrator, and self-consistency samples.

A purely random mock would not let us assert on correctness. A mock that
always returns "A" would mask tie-breaking and aggregation bugs.

## Decision

`gpqa_cmab.llm.mock.MockLLMClient` is the default provider (`LLM_PROVIDER=mock`)
and behaves as follows:

- Each agent type emits a schema-correct payload (validated by Pydantic in
  tests).
- The chosen answer is taken from the literal token `MOCK_CORRECT_ANSWER=<x>`
  found in the prompt; otherwise it falls back to
  `"ABCD"[sha256(prompt) % 4]` for stable but non-trivial answers.
- The `MOCK_CORRECT_ANSWER` token is appended only by the *mock* prompt
  builders in `agents/` and `experiments/self_consistency`. It must never
  appear in production prompts.

## Consequences

- **Positive**: end-to-end smoke test passes deterministically. Coverage
  exceeds the 80% target for core modules without a network call.
- **Positive**: factorial tests can assert `all(row.correct)` and detect
  regressions.
- **Negative**: the mock cannot evaluate research claims. The README and
  reports state this explicitly.
- **Negative**: self-consistency tie-breaking is exercised but not stress-
  tested in mock mode, because the seeded hash collapses identical prompts.
  Real-mode runs must validate tie-breaking with multiple seeds.

## Alternatives considered

- **No mock provider**: rejected — would break the offline guarantee.
- **Random mock**: rejected — would make tests flaky.
- **Recorded VCR-style fixtures**: rejected for the MVP as overkill;
  revisit if real-provider integration tests are added later.
