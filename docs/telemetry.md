# LLM Boundary and Telemetry

All LLM I/O passes through three layers, in order:

1. **Provider abstraction** — `gpqa_cmab.llm.base.LLMClient` (ABC) with a
   `complete(LLMRequest) -> LLMResponse` contract.
2. **Validation + retry wrapper** — `gpqa_cmab.json_utils.complete_validated()`.
   Every agent runner uses it instead of calling `client.complete()` directly.
3. **Telemetry recorder** — `gpqa_cmab.telemetry.TelemetryLogger`, plus the
   aggregation helper `aggregate_usage()`.

## Provider abstraction

```python
class LLMClient(ABC):
    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse: ...
```

Implementations live under `src/gpqa_cmab/llm/`:

- `MockLLMClient` — deterministic, no network. Pulls a gold answer out of
  `MOCK_CORRECT_ANSWER=...` if present in the prompt; otherwise hashes the
  prompt with SHA-256 to pick a stable answer. Each agent type returns a
  schema-correct mock payload.
- `OpenAIClient` — uses the official `openai` SDK. Requires
  `LLM_PROVIDER=openai` and `OPENAI_API_KEY`. Records the provider's reported
  `usage` directly; if usage is missing it is marked
  `usage.estimated = True`.

Adding a new provider means subclassing `LLMClient` and wiring it into
`gpqa_cmab.cli.make_client`. Do not let provider-specific code leak into the
agents or the telemetry layer.

## JSON validation and retries

`complete_validated(client, request, model_type, *, telemetry, record_kwargs,
max_retries=2)` is the canonical way to call an LLM:

1. Logs `llm_request_start` with agent / question / model / attempt.
2. Calls `client.complete()`.
3. On API exception: records a `success=False` telemetry row with
   `error_type`/`error_message`, then re-raises.
4. Strips Markdown code fences (` ```json … ``` `) from the response.
5. Validates against the target Pydantic model.
6. On JSON or schema error: records a `success=False` row, re-prompts with the
   error appended, and retries up to `max_retries` times.
7. After `max_retries + 1` total attempts, raises `ValueError`.
8. On success, records a final `success=True` row and returns
   `(parsed_model, telemetry_row)`.

This is the **only** acceptable retry path. Direct `json.loads(resp.content)`
calls in agent code are a bug.

## Telemetry schemas

### Per-call (`CallTelemetry`)

```json
{
  "request_id": "uuid",
  "experiment_id": "string",
  "question_id": "string",
  "agent_type": "main|A|B|C|D|self_consistency",
  "subset_id": "string",
  "model": "string",
  "prompt_version": "string",
  "temperature": 0.0,
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
             "estimated": false},
  "latency_ms": 0,
  "timestamp_utc": "ISO-8601",
  "success": true,
  "error_type": null,
  "error_message": null
}
```

`request_id` and `timestamp_utc` are auto-populated by Pydantic field
defaults.

### Per-subset aggregate (`AggregateTelemetry`)

Used inside `FactorialResult.usage` to roll up the four subagent calls plus the
main integrator call for one (question, subset) pair. The factorial runner
calls `telemetry.aggregate_usage()` once per row.

### Logging hooks

`complete_validated` emits structured `logging.info` events before and after
every API call. Configure verbosity through `LOG_LEVEL` (see
[.env.example](../.env.example)). When `LOG_LEVEL=DEBUG`, the logger also
captures retry prompts on validation failure.

## Cost accounting rules

- Use the provider's reported usage. The MVP does not estimate tokens unless
  the provider omits usage; in that case `usage.estimated` is `True`.
- Costs are reported in tokens, not dollars. The brief asks for cost-per-correct
  in token units; conversion to USD belongs in a future post-processing step.
- `metrics.utility` normalizes against the average all-four token total so the
  utility is scale-free across model choices.
