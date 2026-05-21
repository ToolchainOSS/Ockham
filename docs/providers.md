# LLM Providers and Vendor Neutrality

This MVP is vendor neutral. Any provider that exposes an OpenAI-API-compatible
chat completions endpoint can drive the pipeline without changing code.

## How the abstraction works

```
agents/* ──► json_utils.complete_validated ──► LLMClient.complete ──► provider SDK
```

Two concrete clients live in
[`src/gpqa_cmab/llm/openai_compatible.py`](../src/gpqa_cmab/llm/openai_compatible.py):

- `OpenAICompatibleClient` — wraps `openai.OpenAI(...)`. Any vendor that
  speaks the OpenAI Chat Completions schema can be reached by setting
  `OPENAI_BASE_URL` (or `LLM_BASE_URL`). This covers OpenAI itself, Together,
  Groq, OpenRouter, Anyscale, Fireworks, DeepSeek, xAI, Mistral, Perplexity,
  LM Studio, local vLLM, local Ollama, and others.
- `AzureOpenAIClient` — wraps `openai.AzureOpenAI(...)`. Azure exposes the
  same schema but uses deployment names and an API version; treat
  `request.model` as the deployment name.

The mock provider (`MockLLMClient`) is always available for offline runs.

## Selecting a provider

`LLM_PROVIDER` controls dispatch in `gpqa_cmab.cli.make_client`. All recognized
values:

| Value | Routes to | Notes |
|---|---|---|
| `mock` | `MockLLMClient` | Default; no network. |
| `openai` | `OpenAICompatibleClient` | Defaults to `https://api.openai.com/v1`. |
| `openai_compatible`, `openai-compatible`, `compatible` | `OpenAICompatibleClient` | Identical behavior; clearer naming for non-OpenAI vendors. |
| `vllm`, `ollama`, `lmstudio`, `local` | `OpenAICompatibleClient` | Convenience aliases for self-hosted servers. |
| `together`, `togetherai`, `groq`, `openrouter`, `anyscale`, `fireworks`, `deepseek`, `xai`, `mistral`, `perplexity` | `OpenAICompatibleClient` | Convenience aliases; you still need to set `OPENAI_BASE_URL`. |
| `azure_openai`, `azure-openai`, `azure` | `AzureOpenAIClient` | Requires `AZURE_OPENAI_*` env vars. |

The alias list is documentation, not enforcement: any of these names produce a
client configured purely from environment variables.

## Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `LLM_PROVIDER` | dispatch | One of the names above. |
| `MAIN_MODEL`, `SUBAGENT_MODEL`, `SELF_CONSISTENCY_MODEL` | every command | Model IDs (or Azure deployment names). |
| `OPENAI_API_KEY` | OpenAI-compatible | Primary API key. |
| `OPENAI_API_KEYS` | OpenAI-compatible | Multiple equivalent keys for RPM load balancing (comma/whitespace separated). Takes precedence over `OPENAI_API_KEY`. See [Load-balancing across multiple keys](#load-balancing-across-multiple-keys). |
| `OPENAI_KEY_COOLDOWN_S` | OpenAI-compatible | Fallback cooldown (seconds) when the server gives no retry hint (default 30). |
| `OPENAI_MAX_RETRIES` | OpenAI-compatible | Max 429s tolerated per request before raising (default 6). |
| `OPENAI_MAX_WAIT_S` | OpenAI-compatible | Total sleep budget per request while all keys are parked (default 120). |
| `LLM_API_KEY` | OpenAI-compatible | Fallback when `OPENAI_API_KEY` is unset (useful for self-hosted setups). |
| `OPENAI_BASE_URL` / `LLM_BASE_URL` | OpenAI-compatible | Endpoint URL override. |
| `OPENAI_ORGANIZATION` | OpenAI | Optional OpenAI organization ID. |
| `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` | Azure | Required for Azure. |
| `LLM_DEFAULT_HEADERS` | both | JSON object or `key=value,key=value` string. Useful for OpenRouter referer headers, custom auth, etc. |
| `LLM_TIMEOUT_S` | both | Per-request timeout (seconds). |

If no API key is set for a self-hosted server, the client falls back to the
sentinel string `"not-needed"` so that `openai.OpenAI(...)` does not reject
construction.

## Provider recipes

These are illustrative; verify the current base URLs with each vendor.

### OpenAI

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export MAIN_MODEL=gpt-4o-mini
export SUBAGENT_MODEL=gpt-4o-mini
```

### Azure OpenAI

```bash
export LLM_PROVIDER=azure_openai
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_API_VERSION=2024-06-01
export MAIN_MODEL=gpt-4o-mini-deployment
export SUBAGENT_MODEL=gpt-4o-mini-deployment
```

### Together

```bash
export LLM_PROVIDER=together
export OPENAI_API_KEY=$TOGETHER_API_KEY
export OPENAI_BASE_URL=https://api.together.xyz/v1
export MAIN_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
export SUBAGENT_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
```

### Groq

```bash
export LLM_PROVIDER=groq
export OPENAI_API_KEY=$GROQ_API_KEY
export OPENAI_BASE_URL=https://api.groq.com/openai/v1
export MAIN_MODEL=llama-3.1-70b-versatile
```

### OpenRouter

```bash
export LLM_PROVIDER=openrouter
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export LLM_DEFAULT_HEADERS='HTTP-Referer=https://example.com,X-Title=gpqa-cmab'
export MAIN_MODEL=openai/gpt-4o-mini
```

### DeepSeek

```bash
export LLM_PROVIDER=deepseek
export OPENAI_API_KEY=$DEEPSEEK_API_KEY
export OPENAI_BASE_URL=https://api.deepseek.com/v1
export MAIN_MODEL=deepseek-chat
```

### Local vLLM

```bash
# vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
export LLM_PROVIDER=vllm
export OPENAI_BASE_URL=http://localhost:8000/v1
export MAIN_MODEL=meta-llama/Llama-3.1-8B-Instruct
export SUBAGENT_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

### Local Ollama

```bash
# ollama serve  # then `ollama pull llama3.1`
export LLM_PROVIDER=ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
export MAIN_MODEL=llama3.1
export SUBAGENT_MODEL=llama3.1
```

## Vendor-neutrality invariants

The codebase enforces these:

- **No provider-specific imports outside `gpqa_cmab.llm.openai_compatible`.**
  Agents, experiments, telemetry, metrics, and reporting all consume the
  abstract `LLMClient` interface. Grep-friendly: `grep -R "from openai" src/`
  should only match the compatibility module.
- **No provider-specific fields in `LLMRequest` / `LLMResponse`.** The
  abstraction speaks model, temperature, prompt, usage. Provider responses
  are stashed under `LLMResponse.raw_response` only.
- **Token accounting uses provider-reported usage when available** and falls
  back to `usage.estimated = True` otherwise. Many OpenAI-compatible vendors
  populate the same `usage` block; those that do not will trigger the
  estimated flag.
- **Configuration lives in environment variables**, not in code. New
  providers should be reachable by changing `.env`, not by editing Python.

## Adding a new provider

In most cases you do not need to add code. If the provider speaks the OpenAI
Chat Completions schema, set `OPENAI_BASE_URL`, optionally add a friendly
alias to `_OPENAI_COMPATIBLE_ALIASES` in
[`cli.py`](../src/gpqa_cmab/cli.py), and document a recipe here.

If the provider uses a non-OpenAI schema (Anthropic Messages, Google
GenerativeLanguage, Bedrock Converse, etc.), add a new subclass of
`gpqa_cmab.llm.base.LLMClient` and translate the request/response in that
class. Register it in `make_client`. Keep the translation isolated to the
new module.

## Troubleshooting

- **Missing usage data**: the `usage.estimated` flag will be `True` and
  token counts will be zero. Either upgrade the provider or live with the
  caveat in reports.
- **401/403 from a self-hosted server**: ensure either `OPENAI_API_KEY` or
  `LLM_API_KEY` is set; even `"not-needed"` works for many servers but
  must be present.
- **OpenRouter "missing referer"**: set `LLM_DEFAULT_HEADERS` with
  `HTTP-Referer` and `X-Title`.
- **Azure 404**: confirm that `MAIN_MODEL` is the *deployment* name, not the
  model family.

## Load-balancing across multiple keys

Most OpenAI-compatible providers enforce a per-key requests-per-minute (RPM)
limit. A factorial sweep across 16 subsets fires 20 calls per question, which
saturates a single low-tier key quickly. The OpenAI-compatible client
supports a built-in **API key pool** that round-robins requests across
multiple **equivalent** keys and automatically rotates away from any key
that returns a 429 `RateLimitError`.

```bash
# .env
OPENAI_API_KEYS=sk-aaa...,sk-bbb...,sk-ccc...
OPENAI_KEY_COOLDOWN_S=30
```

Behaviour:

- Requests are dispatched round-robin across the pool, so 3 keys give ~3×
  the effective RPM ceiling.
- On a 429 from key *i*, the pool extracts the retry delay from:
  1. The `Retry-After` HTTP response header (RFC 7231); else
  2. The provider's error body (e.g. Groq/Together's "Please try again in
     2.4s" or "500ms"); else
  3. The configured `OPENAI_KEY_COOLDOWN_S` fallback.

  Key *i* is parked for that delay and the request is retried on the next
  available key within the same `complete()` call.
- When **every** key is simultaneously parked (the common case for a
  single-key pool, or for multiple keys sharing an org-wide TPM bucket),
  the pool sleeps until the soonest key becomes free and retries —
  bounded by `OPENAI_MAX_RETRIES` (default 6) and `OPENAI_MAX_WAIT_S`
  (default 120). When the budget is exhausted the last `RateLimitError`
  is re-raised so the caller (and telemetry) sees the failure honestly.
- Keys are deduplicated while preserving order.
- The pool is thread-safe (uses a `threading.Lock`) so it is safe under
  future parallel sweeps.

Requirements: all keys MUST belong to the same provider, organization, and
have access to the same models. Do NOT mix keys with different model
allow-lists. The client does not detect that mismatch; you would see
sporadic 404s instead of 429s.

You can also pass keys explicitly:

```python
OpenAICompatibleClient(api_keys=["sk-aaa", "sk-bbb"])
```

The constructor argument takes precedence over both `OPENAI_API_KEYS` and
`OPENAI_API_KEY`.
