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


_VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def _resolve_reasoning_effort(value: str | None) -> str | None:
    """Normalize and validate a reasoning effort value.

    Modern OpenAI reasoning models (gpt-5.x, o-series) plus several
    OpenAI-compatible providers accept reasoning effort with values from the
    set: ``none | minimal | low | medium | high | xhigh``. Some models only
    accept a subset; the API will reject unsupported values at request time.
    Empty / None disables reasoning configuration entirely.
    """
    if value is None:
        return None
    stripped = value.strip().lower()
    if not stripped:
        return None
    if stripped not in _VALID_REASONING_EFFORTS:
        raise ValueError(
            f"Invalid REASONING_EFFORT={value!r}. "
            f"Expected one of: {sorted(_VALID_REASONING_EFFORTS)}."
        )
    return stripped


def _resolve_max_output_tokens(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = int(stripped)
    except ValueError as exc:
        raise ValueError(f"Invalid MAX_OUTPUT_TOKENS={value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"MAX_OUTPUT_TOKENS must be positive (got {parsed}).")
    return parsed


def _resolve_use_responses_api(
    explicit: str | None,
    *,
    reasoning_effort: str | None,
    base_url: str | None,
) -> bool:
    """Decide whether to call `client.responses.create` instead of chat.

    OpenAI recommends the Responses API for reasoning models (better
    intelligence and cleaner API). Most other OpenAI-compatible providers do
    NOT implement /responses yet, so the default policy is:

    - Explicit env override (`LLM_USE_RESPONSES_API=true|false`) wins.
    - Else: auto-enable when reasoning_effort is set AND the endpoint is
      OpenAI's official one (or unset, which defaults to OpenAI).
    - Else: stay on chat completions.
    """
    if explicit is not None:
        stripped = explicit.strip().lower()
        if stripped in {"1", "true", "yes", "on"}:
            return True
        if stripped in {"0", "false", "no", "off", ""}:
            return False
        raise ValueError(
            f"Invalid LLM_USE_RESPONSES_API={explicit!r}. Expected one of: true, false."
        )
    if reasoning_effort is None:
        return False
    if base_url is None:
        return True
    return "api.openai.com" in base_url


def _extract_responses_content(response: Any) -> str:
    """Pull assistant text out of an OpenAI Responses API result."""
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if isinstance(value, str):
                chunks.append(value)
    return "".join(chunks)


def _build_llm_response(
    response: Any, latency_ms: int, *, use_responses_api: bool
) -> LLMResponse:
    """Normalize a Chat-Completions OR Responses payload into our LLMResponse."""
    usage = getattr(response, "usage", None)
    if usage is None:
        usage_model = Usage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0, estimated=True
        )
    else:
        # Chat completions: prompt_tokens / completion_tokens / total_tokens.
        # Responses:        input_tokens / output_tokens / total_tokens.
        prompt_tokens = (
            getattr(usage, "prompt_tokens", None)
            if getattr(usage, "prompt_tokens", None) is not None
            else getattr(usage, "input_tokens", 0)
        )
        completion_tokens = (
            getattr(usage, "completion_tokens", None)
            if getattr(usage, "completion_tokens", None) is not None
            else getattr(usage, "output_tokens", 0)
        )
        total_tokens = getattr(usage, "total_tokens", 0)
        reasoning_tokens = 0
        details = getattr(usage, "output_tokens_details", None) or getattr(
            usage, "completion_tokens_details", None
        )
        if details is not None:
            reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        usage_model = Usage(
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            total_tokens=int(total_tokens or 0),
            reasoning_tokens=reasoning_tokens,
            estimated=False,
        )
    if use_responses_api:
        content = _extract_responses_content(response)
    else:
        content = response.choices[0].message.content or ""
    return LLMResponse(
        content=content,
        usage=usage_model,
        latency_ms=latency_ms,
        raw_response=response.model_dump(mode="json")
        if hasattr(response, "model_dump")
        else None,
    )


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
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        use_responses_api: bool | None = None,
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
        self._reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else _resolve_reasoning_effort(os.environ.get("REASONING_EFFORT"))
        )
        self._max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else _resolve_max_output_tokens(os.environ.get("MAX_OUTPUT_TOKENS"))
        )
        if use_responses_api is None:
            self._use_responses_api = _resolve_use_responses_api(
                os.environ.get("LLM_USE_RESPONSES_API"),
                reasoning_effort=self._reasoning_effort,
                base_url=self._base_url,
            )
        else:
            self._use_responses_api = use_responses_api

    @property
    def base_url(self) -> str | None:
        return self._base_url

    @property
    def reasoning_effort(self) -> str | None:
        return self._reasoning_effort

    @property
    def use_responses_api(self) -> bool:
        return self._use_responses_api

    @property
    def max_output_tokens(self) -> int | None:
        return self._max_output_tokens

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        if self._use_responses_api:
            response = self._invoke_responses_api(request)
        else:
            response = self._invoke_chat_completions(request)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return self._build_llm_response(response, latency_ms)

    def _invoke_chat_completions(self, request: LLMRequest):
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if self._reasoning_effort is not None:
            # Chat-completions reasoning models accept a flat reasoning_effort
            # parameter and reject non-default temperature.
            kwargs["reasoning_effort"] = self._reasoning_effort
        else:
            kwargs["temperature"] = request.temperature
        if self._max_output_tokens is not None:
            kwargs["max_tokens"] = self._max_output_tokens
        return self._client.chat.completions.create(**kwargs)

    def _invoke_responses_api(self, request: LLMRequest):
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": [{"role": "user", "content": request.prompt}],
        }
        if self._reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}
        if self._max_output_tokens is not None:
            kwargs["max_output_tokens"] = self._max_output_tokens
        return self._client.responses.create(**kwargs)

    @staticmethod
    def _extract_responses_content(response: Any) -> str:
        # The Responses API exposes a convenience `output_text` accessor that
        # concatenates all assistant text outputs.
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text:
            return text
        # Fallback: walk the `output` array and grab text content items.
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if isinstance(value, str):
                    chunks.append(value)
        return "".join(chunks)

    def _build_llm_response(self, response: Any, latency_ms: int) -> LLMResponse:
        return _build_llm_response(
            response, latency_ms, use_responses_api=self._use_responses_api
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
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        use_responses_api: bool | None = None,
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
        self._reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else _resolve_reasoning_effort(os.environ.get("REASONING_EFFORT"))
        )
        self._max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else _resolve_max_output_tokens(os.environ.get("MAX_OUTPUT_TOKENS"))
        )
        # Azure's responses endpoint requires api-version 2025-03+ and is not
        # universally enabled; default OFF unless user opts in explicitly.
        if use_responses_api is None:
            explicit = os.environ.get("LLM_USE_RESPONSES_API")
            self._use_responses_api = (
                _resolve_use_responses_api(
                    explicit,
                    reasoning_effort=self._reasoning_effort,
                    base_url=None,  # treat as non-OpenAI for auto policy
                )
                if explicit is not None
                else False
            )
        else:
            self._use_responses_api = use_responses_api

    @property
    def reasoning_effort(self) -> str | None:
        return self._reasoning_effort

    @property
    def use_responses_api(self) -> bool:
        return self._use_responses_api

    @property
    def max_output_tokens(self) -> int | None:
        return self._max_output_tokens

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        if self._use_responses_api:
            kwargs: dict[str, Any] = {
                "model": request.model,
                "input": [{"role": "user", "content": request.prompt}],
            }
            if self._reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": self._reasoning_effort}
            if self._max_output_tokens is not None:
                kwargs["max_output_tokens"] = self._max_output_tokens
            response = self._client.responses.create(**kwargs)
        else:
            kwargs = {
                "model": request.model,
                "messages": [{"role": "user", "content": request.prompt}],
            }
            if self._reasoning_effort is not None:
                kwargs["reasoning_effort"] = self._reasoning_effort
            else:
                kwargs["temperature"] = request.temperature
            if self._max_output_tokens is not None:
                kwargs["max_tokens"] = self._max_output_tokens
            response = self._client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return _build_llm_response(
            response, latency_ms, use_responses_api=self._use_responses_api
        )


# Backwards-compatible alias retained for existing imports.
OpenAIClient = OpenAICompatibleClient
