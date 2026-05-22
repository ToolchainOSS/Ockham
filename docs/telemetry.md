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
- `OpenAICompatibleClient` — uses the official `openai` SDK and works against
  **any** OpenAI-API-compatible vendor (OpenAI, Together, Groq, OpenRouter,
  Anyscale, Fireworks, DeepSeek, xAI, Mistral, Perplexity, local vLLM, local
  Ollama, …). Endpoint and credentials come from environment variables; see
  [providers.md](providers.md).
- `AzureOpenAIClient` — uses `openai.AzureOpenAI`; `request.model` is the
  Azure *deployment* name.

Provider-specific responses are recorded directly: when `usage` is missing the
client marks `usage.estimated = True`. Provider-specific imports must stay
inside `gpqa_cmab.llm.openai_compatible`; agents and experiments only see the
`LLMClient` ABC.

Adding a new vendor with the same schema means changing `.env`, not Python
code. Adding a new vendor with a different schema means subclassing
`LLMClient` and translating in the new module — see
[providers.md](providers.md#adding-a-new-provider).

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
defaults. Prompt text, response text, request metadata, and raw provider
responses are stored in the trace so the first expensive run can be audited
without re-running it. The trace also stores SHA-256 hashes and character
counts for integrity checks. Known API tokens and other secret environment
values are redacted before text payloads are persisted.

### Per-subset aggregate (`AggregateTelemetry`)

Used inside `FactorialResult.usage` to roll up the four subagent calls plus the
main integrator call for one (question, subset) pair. The factorial runner
calls `telemetry.aggregate_usage()` once per row.

### Logging hooks

`complete_validated` emits structured `logging.info` events before and after
every API call. Commands that issue LLM calls also attach a per-run log file
next to the result artifact, for example:

- `full_factorial_results.jsonl.trace.jsonl` — one row per LLM attempt.
- `full_factorial_results.jsonl.log` — structured runtime log messages.
- `full_factorial_results.jsonl.manifest.json` — run manifest and checksums.

Configure verbosity through `LOG_LEVEL` (see [.env.example](../.env.example)).
Logs intentionally avoid credentials; prompt and response bodies belong in the
trace JSONL, not the human-readable log file.

## Run manifests

Every artifact-producing command writes a manifest with `schema_version = 2`.
The manifest is the audit envelope for paper submission and includes:

- exact CLI `argv` used for the run;
- start/finish timestamps and command status;
- SHA-256, byte size, modified timestamp, and JSONL row counts for inputs,
  outputs, prompts, runtime source files, traces, and log files;
- trace summary totals by agent, model, success/failure count, retry attempt,
  error type, prompt/completion/reasoning tokens, and estimated-usage rows;
- sanitized environment inventory for all supported runtime variables;
- runtime metadata including Python version, platform, current working
  directory, git commit, branch, dirty flag, and porcelain status;
- settings, budget/call caps, and command-specific extras.

Sensitive environment values (`*_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`,
`*_AUTH`, `*_HEADER`) are never written directly. The manifest stores only
whether each variable was set, item counts for multi-key values, lengths, and
SHA-256 fingerprints. The same known-secret redactor is applied to prompt,
response, metadata, error-message, and raw-response fields before trace rows
are persisted.

For an expensive first real run, archive the result JSONL, trace JSONL, log,
manifest, prompt files, git commit, source tree, and the generated
evaluation/report artifacts together. The manifest hashes allow the bundle to
be checked for later drift.

## Cost accounting rules

- Use the provider's reported usage. The MVP does not estimate tokens unless
  the provider omits usage; in that case `usage.estimated` is `True`.
- Costs are reported in tokens, not dollars. The brief asks for cost-per-correct
  in token units; conversion to USD belongs in a future post-processing step.
- `metrics.utility` normalizes against the average all-four token total so the
  utility is scale-free across model choices.
