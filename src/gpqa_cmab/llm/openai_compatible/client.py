"""The concrete ``OpenAICompatibleClient`` and ``AzureOpenAIClient``.

Part of the vendor boundary: the ``openai`` SDK is imported lazily inside the
constructors so the rest of the codebase never depends on it directly.
"""

from __future__ import annotations

import os
import time
from typing import Any

from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.llm.openai_compatible.config import (
    _parse_headers,
    _resolve_api_keys,
    _resolve_cooldown_seconds,
    _resolve_max_output_tokens,
    _resolve_nonneg_float,
    _resolve_positive_int,
    _resolve_reasoning_effort,
    _resolve_timeout,
)
from gpqa_cmab.llm.openai_compatible.key_pool import _KeyPool
from gpqa_cmab.llm.openai_compatible.payload import (
    _build_llm_response,
    _chat_kwargs,
    _resolve_use_responses_api,
    _responses_kwargs,
)
from gpqa_cmab.schemas import LLMRequest, LLMResponse


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
        api_keys: list[str] | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        use_responses_api: bool | None = None,
        key_cooldown_seconds: float | None = None,
        max_retry_attempts: int | None = None,
        max_retry_wait_seconds: float | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised in CI only
            raise RuntimeError(
                "Install gpqa-cmab[openai] to use OpenAICompatibleClient"
            ) from exc

        # Resolve the ordered list of equivalent API keys to load-balance over.
        resolved_keys = _resolve_api_keys(
            explicit_single=api_key,
            explicit_multi=",".join(api_keys) if api_keys else None,
            env_keys=os.environ.get("OPENAI_API_KEYS"),
            env_legacy_key=os.environ.get("OPENAI_API_KEY"),
            env_alt_key=os.environ.get("LLM_API_KEY"),
        )

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

        shared_kwargs: dict[str, Any] = {}
        if resolved_base_url:
            shared_kwargs["base_url"] = resolved_base_url
        if resolved_org:
            shared_kwargs["organization"] = resolved_org
        if resolved_headers:
            shared_kwargs["default_headers"] = resolved_headers
        if resolved_timeout is not None:
            shared_kwargs["timeout"] = resolved_timeout

        clients = [OpenAI(**{**shared_kwargs, "api_key": key}) for key in resolved_keys]
        cooldown = (
            key_cooldown_seconds
            if key_cooldown_seconds is not None
            else _resolve_cooldown_seconds(os.environ.get("OPENAI_KEY_COOLDOWN_S"))
        )
        max_attempts = (
            max_retry_attempts
            if max_retry_attempts is not None
            else _resolve_positive_int(
                os.environ.get("OPENAI_MAX_RETRIES"),
                default=6,
                var_name="OPENAI_MAX_RETRIES",
            )
        )
        max_wait = (
            max_retry_wait_seconds
            if max_retry_wait_seconds is not None
            else _resolve_nonneg_float(
                os.environ.get("OPENAI_MAX_WAIT_S"),
                default=120.0,
                var_name="OPENAI_MAX_WAIT_S",
            )
        )
        self._key_pool = _KeyPool(
            clients,
            cooldown_seconds=cooldown,
            max_attempts=max_attempts,
            max_wait_seconds=max_wait,
        )
        # Keep `_client` pointing at the first pool member for backwards
        # compatibility with any external code introspecting it (tests, etc.).
        self._client = clients[0]
        self._api_key_count = len(resolved_keys)
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

    @property
    def api_key_count(self) -> int:
        """Number of equivalent keys available for load-balancing."""
        return self._api_key_count

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        if self._use_responses_api:
            response = self._key_pool.execute(
                lambda client: self._invoke_responses_api(client, request)
            )
        else:
            response = self._key_pool.execute(
                lambda client: self._invoke_chat_completions(client, request)
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return _build_llm_response(
            response, latency_ms, use_responses_api=self._use_responses_api
        )

    def _invoke_chat_completions(self, client: Any, request: LLMRequest) -> Any:
        return client.chat.completions.create(
            **_chat_kwargs(
                request,
                reasoning_effort=self._reasoning_effort,
                max_output_tokens=self._max_output_tokens,
            )
        )

    def _invoke_responses_api(self, client: Any, request: LLMRequest) -> Any:
        return client.responses.create(
            **_responses_kwargs(
                request,
                reasoning_effort=self._reasoning_effort,
                max_output_tokens=self._max_output_tokens,
            )
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
            response = self._client.responses.create(
                **_responses_kwargs(
                    request,
                    reasoning_effort=self._reasoning_effort,
                    max_output_tokens=self._max_output_tokens,
                )
            )
        else:
            response = self._client.chat.completions.create(
                **_chat_kwargs(
                    request,
                    reasoning_effort=self._reasoning_effort,
                    max_output_tokens=self._max_output_tokens,
                )
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return _build_llm_response(
            response, latency_ms, use_responses_api=self._use_responses_api
        )


# Backwards-compatible alias retained for existing imports.
OpenAIClient = OpenAICompatibleClient
