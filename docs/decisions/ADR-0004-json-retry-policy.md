# ADR-0004: Centralize JSON retries and API error logging at the LLM boundary

- Status: Accepted
- Date: 2026-05-21

## Context

The original agent runners called `client.complete()` and `json.loads()`
directly. Any malformed response (extra prose, Markdown fence, near-miss
JSON) crashed the run. API errors crashed before any telemetry was recorded.
The brief mandates: validate every output, retry malformed JSON at most
twice, mark failures explicitly, and log every API call before and after
execution.

Per-agent retry code would duplicate the contract and drift.

## Decision

Introduce `gpqa_cmab.json_utils.complete_validated()` as the single LLM call
path used by every agent. The helper:

1. Logs `llm_request_start` before the call.
2. Catches API exceptions, records a `success=False` telemetry row, and
   re-raises.
3. Strips Markdown code fences from the response.
4. Validates against the target Pydantic schema.
5. Retries up to two times on `json.JSONDecodeError` or `ValidationError`,
   re-prompting the model with the error.
6. Records every attempt in telemetry (`success=False` for failed attempts,
   `success=True` for the final accepted one).
7. Raises `ValueError` if all attempts fail.

Agents pass `record_kwargs` describing the call (agent type, subset, model,
prompt version, temperature) and the helper threads those into telemetry.

## Consequences

- **Positive**: one place to enforce the brief's guardrails. Tests cover
  retry-then-succeed, API failure, and retry exhaustion.
- **Positive**: telemetry is complete even on failure paths, which is
  required for cost accounting.
- **Negative**: agents must use the helper; bypassing it is now a bug.
  Documented in [development.md](../development.md) and AGENTS.md.

## Alternatives considered

- **Decorator on `LLMClient.complete`**: harder to thread telemetry through
  and conflates provider abstraction with validation.
- **Per-agent retry loops**: rejected as the source of drift the brief warns
  against.
