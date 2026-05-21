"""OpenAI-API-compatible LLM client.

This module is vendor neutral. The same `OpenAICompatibleClient` works
against any provider that exposes an OpenAI-compatible chat completions
endpoint: OpenAI, Azure OpenAI, Together, Groq, OpenRouter, Anyscale,
Fireworks, DeepSeek, xAI, Mistral, local vLLM, local Ollama, etc.

Provider selection is driven by environment variables. See
``docs/providers.md`` for full configuration examples.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.schemas import LLMRequest, LLMResponse, Usage


def _parse_headers(raw: str | None) -> dict[str, str] | None:
    """Parse default headers from env. Accepts JSON object or k=v,k=v list."""
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("{"):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in LLM_DEFAULT_HEADERS: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("LLM_DEFAULT_HEADERS JSON must be an object.")
        return {str(k): str(v) for k, v in value.items()}
    headers: dict[str, str] = {}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise ValueError(
                f"Invalid LLM_DEFAULT_HEADERS entry {pair!r}; expected key=value."
            )
        key, value = pair.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers or None


def _resolve_timeout(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid LLM_TIMEOUT_S value: {value!r}") from exc


class OpenAICompatibleClient(LLMClient):
    """Generic OpenAI-API-compatible client.

    Reads configuration from constructor arguments first, then environment
    variables. The constructor is the test surface; env vars are the
    production surface.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised in CI only
            raise RuntimeError(
                "Install gpqa-cmab[openai] to use OpenAICompatibleClient"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        # Many self-hosted servers do not require a key; the OpenAI SDK does.
        # Accept a sentinel so users can run against local vLLM/Ollama.
        if not resolved_key:
            resolved_key = os.environ.get("LLM_API_KEY") or "not-needed"

        resolved_base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
        )
        resolved_org = organization or os.environ.get("OPENAI_ORGANIZATION")
        resolved_headers = default_headers or _parse_headers(
            os.environ.get("LLM_DEFAULT_HEADERS")
        )
        resolved_timeout = (
            timeout
            if timeout is not None
            else _resolve_timeout(os.environ.get("LLM_TIMEOUT_S"))
        )

        kwargs: dict[str, Any] = {"api_key": resolved_key}
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        if resolved_org:
            kwargs["organization"] = resolved_org
        if resolved_headers:
            kwargs["default_headers"] = resolved_headers
        if resolved_timeout is not None:
            kwargs["timeout"] = resolved_timeout

        self._client = OpenAI(**kwargs)
        self._base_url = resolved_base_url

    @property
    def base_url(self) -> str | None:
        return self._base_url

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=request.model,
            temperature=request.temperature,
            messages=[{"role": "user", "content": request.prompt}],
        )
        usage = getattr(response, "usage", None)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return LLMResponse(
            content=response.choices[0].message.content or "",
            usage=Usage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                estimated=usage is None,
            ),
            latency_ms=latency_ms,
            raw_response=response.model_dump(mode="json"),
        )


class AzureOpenAIClient(LLMClient):
    """Azure OpenAI client.

    Azure exposes the same chat completions schema but uses deployment names
    instead of model IDs and requires an API version. Configure via
    ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``, and
    ``AZURE_OPENAI_API_VERSION``. The ``request.model`` field is treated as
    the Azure deployment name.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover - exercised in CI only
            raise RuntimeError(
                "Install gpqa-cmab[openai] to use AzureOpenAIClient"
            ) from exc

        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        resolved_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        resolved_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        if not resolved_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for azure_openai.")
        if not resolved_version:
            raise ValueError("AZURE_OPENAI_API_VERSION is required for azure_openai.")

        resolved_headers = default_headers or _parse_headers(
            os.environ.get("LLM_DEFAULT_HEADERS")
        )
        resolved_timeout = (
            timeout
            if timeout is not None
            else _resolve_timeout(os.environ.get("LLM_TIMEOUT_S"))
        )

        kwargs: dict[str, Any] = {
            "api_key": resolved_key,
            "azure_endpoint": resolved_endpoint,
            "api_version": resolved_version,
        }
        if resolved_headers:
            kwargs["default_headers"] = resolved_headers
        if resolved_timeout is not None:
            kwargs["timeout"] = resolved_timeout
        self._client = AzureOpenAI(**kwargs)

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=request.model,
            temperature=request.temperature,
            messages=[{"role": "user", "content": request.prompt}],
        )
        usage = getattr(response, "usage", None)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return LLMResponse(
            content=response.choices[0].message.content or "",
            usage=Usage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                estimated=usage is None,
            ),
            latency_ms=latency_ms,
            raw_response=response.model_dump(mode="json"),
        )


# Backwards-compatible alias retained for existing imports.
OpenAIClient = OpenAICompatibleClient
