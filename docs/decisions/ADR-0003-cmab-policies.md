# ADR-0003: Ship both Super-arm TS and a structured CMAB

- Status: Accepted
- Date: 2026-05-21

## Context

The brief asks for both a simple super-arm bandit and a structured CMAB that
shares information across subsets. With only four subagents and 16 subsets,
the simple sampler works. The point of the structured policy is to validate
the methodology under a model that *would* scale, not to win on the
four-subagent instance.

## Decision

Implement two policies behind the same `Learner` shape:

- `SuperArmThompsonSampler` (Beta-Bernoulli per subset) as a baseline.
- `StructuredCMAB` (online logistic regression over singletons, pairwise
  interactions, `num_subagents`, and `estimated_token_cost`) with a
  TS-style uncertainty bonus.

Both consume the same `FactorialResult` rows under the partial-information
replay protocol. The CLI `--policy` flag selects between them.

## Consequences

- **Positive**: regression tests pin down both code paths, and reports can
  compare them.
- **Positive**: when subagent count grows, only the structured CMAB stays
  practical, and the API is already in place.
- **Negative**: two policies, two surfaces, slightly more code to maintain.
  Acceptable.

## Alternatives considered

- **Linear UCB instead of structured TS**: viable but less consistent with
  the brief's wording and the surrounding super-arm TS approach.
- **Full Bayesian logistic regression**: rejected as too heavyweight for an
  MVP; the online approximation suffices.
