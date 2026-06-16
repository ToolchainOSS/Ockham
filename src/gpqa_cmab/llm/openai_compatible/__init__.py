"""OpenAI-API-compatible LLM client (vendor-neutral package).

The same ``OpenAICompatibleClient`` works against any provider that exposes an
OpenAI-compatible chat completions endpoint: OpenAI, Azure OpenAI, Together,
Groq, OpenRouter, Anyscale, Fireworks, DeepSeek, xAI, Mistral, local vLLM,
local Ollama, etc. Provider selection is driven by environment variables; see
``docs/providers.md`` for full configuration examples.

This package is the ``openai`` SDK boundary: vendor imports live only in
``key_pool`` and ``client``. The public surface re-exported here is stable.
"""

from __future__ import annotations

from gpqa_cmab.llm.openai_compatible.client import (
    AzureOpenAIClient,
    OpenAIClient,
    OpenAICompatibleClient,
)
from gpqa_cmab.llm.openai_compatible.config import _parse_headers
from gpqa_cmab.llm.openai_compatible.key_pool import _extract_retry_delay

__all__ = [
    "AzureOpenAIClient",
    "OpenAIClient",
    "OpenAICompatibleClient",
    "_extract_retry_delay",
    "_parse_headers",
]
