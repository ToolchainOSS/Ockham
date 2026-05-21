# ADR-0002: Subagents are independent; no scratch summary

- Status: Accepted
- Date: 2026-05-21

## Context

A natural extension of multi-agent inference is to let one subagent summarize
the question for the others, or to chain subagent outputs. That makes
measurement harder: the contribution of any single subagent becomes
conditional on the upstream summary, and the bandit can no longer treat
subset membership as the unit of intervention.

The brief explicitly forbids a scratch summary and cross-subagent
communication.

## Decision

Every selected subagent receives the *same raw input*: the question plus the
four choices. Subagents do not see each other's outputs. The main integrator
is the only agent that consumes subagent reports, and it sees only the
reports for the *selected* subset.

## Consequences

- **Positive**: subagent runs can be cached per question. The full factorial
  costs `4 + 16 = 20` LLM calls per question instead of `5 × 16 = 80`.
- **Positive**: subagent A/B/C/D contributions are measurable independently;
  the structured CMAB feature vector is well-defined.
- **Positive**: subagents are trivially parallelizable.
- **Negative**: we forfeit potential accuracy gains from subagent
  collaboration. This is on purpose — the research question is *cost*, not
  peak accuracy.

## Alternatives considered

- **Shared scratchpad**: rejected for the reasons above.
- **Pipeline (A → B → C → D)**: rejected; would force serial execution and
  break the bandit's subset-as-unit-of-intervention assumption.
