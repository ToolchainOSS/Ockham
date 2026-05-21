# Architecture Decision Records

ADRs capture significant design choices, the context, the alternatives
considered, and the resulting decision. New ADRs are appended; existing ones
are not edited except for superseding metadata.

## Format

Each ADR is a Markdown file named `ADR-NNNN-short-slug.md`, with a header:

```markdown
# ADR-NNNN: Title

- Status: Accepted | Superseded by ADR-XXXX
- Date: YYYY-MM-DD
```

Followed by sections: **Context**, **Decision**, **Consequences**, and
optionally **Alternatives considered**.

## Index

| ID | Title | Status |
|---|---|---|
| [ADR-0001](ADR-0001-python-only-mvp.md) | Python-only local MVP (no frontend, no web service) | Accepted |
| [ADR-0002](ADR-0002-no-scratch-summary.md) | Subagents are independent; no scratch summary | Accepted |
| [ADR-0003](ADR-0003-cmab-policies.md) | Ship both Super-arm TS and a structured CMAB | Accepted |
| [ADR-0004](ADR-0004-json-retry-policy.md) | Centralize JSON retries and API error logging at the LLM boundary | Accepted |
| [ADR-0005](ADR-0005-mock-provider-default.md) | Mock provider is the default; embeds gold answer for determinism | Accepted |

## When to write an ADR

- A decision changes a module's public contract.
- A trade-off has more than one defensible answer and reviewers may disagree
  later.
- The decision constrains future work (e.g. a schema field is now required).

Routine refactors, performance tweaks, and bug fixes do not need an ADR.
