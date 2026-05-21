"""Backwards-compatible shim. Prefer ``gpqa_cmab.llm.openai_compatible``."""

from gpqa_cmab.llm.openai_compatible import (
    AzureOpenAIClient,
    OpenAIClient,
    OpenAICompatibleClient,
)

__all__ = ["AzureOpenAIClient", "OpenAICompatibleClient", "OpenAIClient"]
