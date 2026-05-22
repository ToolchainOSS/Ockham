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
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.schemas import LLMRequest, LLMResponse, Usage

logger = logging.getLogger(__name__)


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


T = TypeVar("T")


def _parse_env(
    value: str | None,
    parser: Callable[[str], T],
    *,
    var_name: str,
    validator: Callable[[T], None] | None = None,
) -> T | None:
    """Parse an optional env string into a typed value.

    Returns ``None`` for unset / blank input. Raises ``ValueError`` (with the
    env var name) when ``parser`` or ``validator`` rejects the value.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = parser(stripped)
    except ValueError as exc:
        raise ValueError(f"Invalid {var_name}={value!r}: {exc}") from exc
    if validator is not None:
        validator(parsed)
    return parsed


def _parse_env_or(
    value: str | None,
    parser: Callable[[str], T],
    *,
    var_name: str,
    default: T,
    validator: Callable[[T], None] | None = None,
) -> T:
    parsed = _parse_env(value, parser, var_name=var_name, validator=validator)
    return default if parsed is None else parsed


def _require_positive(value: float, *, var_name: str) -> None:
    if value < 1:
        raise ValueError(f"{var_name} must be >= 1 (got {value}).")


def _require_nonneg(value: float, *, var_name: str) -> None:
    if value < 0:
        raise ValueError(f"{var_name} must be >= 0 (got {value}).")


_VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def _resolve_timeout(value: str | None) -> float | None:
    return _parse_env(value, float, var_name="LLM_TIMEOUT_S")


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
    return _parse_env(
        value,
        int,
        var_name="MAX_OUTPUT_TOKENS",
        validator=lambda v: _require_positive(v, var_name="MAX_OUTPUT_TOKENS"),
    )


def _resolve_api_keys(
    explicit_single: str | None,
    explicit_multi: str | None,
    *,
    env_keys: str | None,
    env_legacy_key: str | None,
    env_alt_key: str | None,
) -> list[str]:
    """Build the ordered list of API keys to load-balance across.

    Precedence (first non-empty wins):
      1. Explicit constructor `api_keys=[...]` (handled by caller via
         `explicit_multi` as a pre-comma-joined string).
      2. Explicit constructor `api_key="..."` (single key).
      3. `OPENAI_API_KEYS` env (comma- or whitespace-separated).
      4. `OPENAI_API_KEY` env (single).
      5. `LLM_API_KEY` env (single, used by many self-hosted servers).
      6. Sentinel "not-needed" for keyless local endpoints.

    All keys are assumed to be EQUIVALENT (same model access, same org).
    Duplicates are removed while preserving order.
    """
    candidates: list[str] = []
    if explicit_multi:
        candidates = _split_keys(explicit_multi)
    elif explicit_single:
        candidates = [explicit_single]
    elif env_keys:
        candidates = _split_keys(env_keys)
    elif env_legacy_key:
        candidates = [env_legacy_key]
    elif env_alt_key:
        candidates = [env_alt_key]
    else:
        candidates = ["not-needed"]
    # Dedupe while preserving order; drop empties.
    seen: set[str] = set()
    unique: list[str] = []
    for key in candidates:
        key = key.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    if not unique:
        unique = ["not-needed"]
    return unique


def _split_keys(raw: str) -> list[str]:
    """Split a multi-key string by comma, newline, or whitespace."""
    parts: list[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.extend(part for part in chunk.split() if part)
    return parts


def _resolve_cooldown_seconds(value: str | None, *, default: float = 30.0) -> float:
    return _parse_env_or(
        value,
        float,
        var_name="OPENAI_KEY_COOLDOWN_S",
        default=default,
        validator=lambda v: _require_nonneg(v, var_name="OPENAI_KEY_COOLDOWN_S"),
    )


def _resolve_positive_int(value: str | None, *, default: int, var_name: str) -> int:
    return _parse_env_or(
        value,
        int,
        var_name=var_name,
        default=default,
        validator=lambda v: _require_positive(v, var_name=var_name),
    )


def _resolve_nonneg_float(value: str | None, *, default: float, var_name: str) -> float:
    return _parse_env_or(
        value,
        float,
        var_name=var_name,
        default=default,
        validator=lambda v: _require_nonneg(v, var_name=var_name),
    )


class _KeyPool:
    """Round-robin pool of OpenAI clients that share equivalent API keys.

    On a `RateLimitError` (HTTP 429) the offending key is parked for a
    delay window derived from the server's hint (``Retry-After`` header or
    "try again in Xs" message body); if no hint is provided we fall back to
    ``cooldown_seconds``. The next available key is then used.

    When *all* keys are parked (the common case for a single-key pool, or
    for multiple keys that share an org-wide TPM bucket), the pool sleeps
    until the soonest key becomes free and retries — bounded by
    ``max_attempts`` and ``max_wait_seconds``.

    Designed for the OpenAI Python SDK's ``OpenAI`` / ``AzureOpenAI`` clients
    but the type is generic; we only call ``.chat.completions.create`` and
    ``.responses.create`` on each member.
    """

    # Heuristic floor so we never sleep less than this between retries after
    # a 429 (avoids hammering the provider when its hint is essentially zero).
    _MIN_WAIT_SECONDS = 0.5

    def __init__(
        self,
        clients: list[Any],
        *,
        cooldown_seconds: float = 30.0,
        max_attempts: int = 6,
        max_wait_seconds: float = 120.0,
        rate_limit_exception: type[BaseException] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not clients:
            raise ValueError("_KeyPool requires at least one client.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1.")
        if max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must be >= 0.")
        self._clients = clients
        self._cooldown_seconds = cooldown_seconds
        self._max_attempts = max_attempts
        self._max_wait_seconds = max_wait_seconds
        self._cooldown_until = [0.0] * len(clients)
        self._cursor = 0
        self._lock = threading.Lock()
        self._clock = clock
        self._sleeper = sleeper
        if rate_limit_exception is None:
            rate_limit_exception = _import_rate_limit_exception()
        self._rate_limit_exception = rate_limit_exception

    @property
    def size(self) -> int:
        return len(self._clients)

    def _next_available(self) -> tuple[int, Any] | None:
        now = self._clock()
        n = len(self._clients)
        with self._lock:
            for offset in range(n):
                index = (self._cursor + offset) % n
                if self._cooldown_until[index] <= now:
                    self._cursor = (index + 1) % n
                    return index, self._clients[index]
        return None

    def _seconds_until_next_free(self) -> float:
        with self._lock:
            soonest = min(self._cooldown_until)
        return max(0.0, soonest - self._clock())

    def _park(self, index: int, seconds: float) -> None:
        with self._lock:
            self._cooldown_until[index] = self._clock() + max(seconds, 0.0)

    def execute(self, call: Callable[[Any], Any]) -> Any:
        """Run ``call(client)`` on a free key, rotating + waiting on 429s."""
        last_error: BaseException | None = None
        attempts = 0
        deadline = self._clock() + self._max_wait_seconds

        while attempts < self._max_attempts:
            picked = self._next_available()
            if picked is not None:
                index, client = picked
                try:
                    return call(client)
                except self._rate_limit_exception as exc:
                    last_error = exc
                    attempts += 1
                    hinted = _extract_retry_delay(exc)
                    delay = hinted if hinted is not None else self._cooldown_seconds
                    delay = max(delay, self._MIN_WAIT_SECONDS)
                    self._park(index, delay)
                    logger.warning(
                        "openai_key_rate_limited index=%d delay_s=%.2f "
                        "attempt=%d/%d pool_size=%d hinted=%s",
                        index,
                        delay,
                        attempts,
                        self._max_attempts,
                        len(self._clients),
                        "yes" if hinted is not None else "no",
                    )
                    continue

            # No key is currently free. Sleep until the soonest one is, then
            # retry — provided we still have budget.
            wait_for = self._seconds_until_next_free()
            now = self._clock()
            if wait_for <= 0:
                # Race: a key just became free. Loop and pick it up.
                continue
            if now + wait_for > deadline:
                logger.error(
                    "openai_pool_exhausted reason=budget all_keys_cooling "
                    "wait_s=%.2f budget_remaining_s=%.2f pool_size=%d",
                    wait_for,
                    max(0.0, deadline - now),
                    len(self._clients),
                )
                break
            logger.info(
                "openai_pool_waiting_all_keys_cooling sleep_s=%.2f pool_size=%d",
                wait_for,
                len(self._clients),
            )
            self._sleeper(wait_for)

        if last_error is not None:
            raise last_error
        raise RuntimeError(
            f"All {len(self._clients)} OpenAI API keys are cooling down."
        )


# Regex matches Groq-style messages: "Please try again in 2.4s" or
# "try again in 1m30s" / "try again in 500ms". We keep the parser narrow
# (seconds + minutes + milliseconds) since OpenAI/Groq/Together all use
# similar phrasing.
_RETRY_BODY_PATTERNS = (
    re.compile(r"try\s+again\s+in\s+([\d.]+)\s*ms", re.IGNORECASE),
    re.compile(r"try\s+again\s+in\s+(?:(\d+)\s*m)?\s*([\d.]+)\s*s", re.IGNORECASE),
)


def _extract_retry_delay(exc: BaseException) -> float | None:
    """Extract a retry delay (seconds) from a RateLimitError.

    Order of preference:
      1. ``Retry-After`` HTTP header on the response (RFC 7231).
      2. Provider-specific hint embedded in the error body
         (Groq/Together: "Please try again in X.Ys").

    Returns ``None`` if no hint is available; callers should then fall back
    to their configured cooldown.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None) or {}
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:  # noqa: BLE001 - mapping-like best-effort
            value = None
        if value:
            try:
                parsed = float(value)
                if parsed >= 0:
                    return parsed
            except (TypeError, ValueError):
                pass

    # Fall back to scraping the error body / message.
    message_sources: list[str] = []
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            message_sources.append(err["message"])
        elif isinstance(body.get("message"), str):
            message_sources.append(body["message"])
    message_sources.append(str(exc))

    for text in message_sources:
        # ms first to avoid the seconds regex matching it as "0s" by accident.
        ms_match = _RETRY_BODY_PATTERNS[0].search(text)
        if ms_match:
            try:
                return float(ms_match.group(1)) / 1000.0
            except ValueError:
                pass
        s_match = _RETRY_BODY_PATTERNS[1].search(text)
        if s_match:
            try:
                minutes = float(s_match.group(1) or 0)
                seconds = float(s_match.group(2))
                return minutes * 60.0 + seconds
            except ValueError:
                pass
    return None


def _import_rate_limit_exception() -> type[BaseException]:
    """Import the OpenAI SDK's RateLimitError lazily.

    Falls back to a sentinel exception class that nothing will raise so the
    pool effectively disables rotation when the SDK is unavailable (tests
    typically inject a stub).
    """
    try:
        from openai import RateLimitError  # type: ignore

        return RateLimitError  # noqa: F401
    except Exception:  # pragma: no cover - openai installed in CI

        class _UnreachableRateLimit(Exception):
            pass

        return _UnreachableRateLimit


def _build_openai_clients_for_pool(
    api_keys: list[str], shared_kwargs: dict[str, Any]
) -> list[Any]:
    """Construct one OpenAI client per key, sharing all other settings."""
    from openai import OpenAI

    return [OpenAI(**{**shared_kwargs, "api_key": key}) for key in api_keys]


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
        cached_prompt_tokens = 0
        prompt_audio_tokens = 0
        completion_audio_tokens = 0
        prompt_details = getattr(usage, "input_tokens_details", None) or getattr(
            usage, "prompt_tokens_details", None
        )
        if prompt_details is not None:
            cached_prompt_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
            prompt_audio_tokens = int(getattr(prompt_details, "audio_tokens", 0) or 0)
        details = getattr(usage, "output_tokens_details", None) or getattr(
            usage, "completion_tokens_details", None
        )
        if details is not None:
            reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
            completion_audio_tokens = int(getattr(details, "audio_tokens", 0) or 0)
        usage_model = Usage(
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            total_tokens=int(total_tokens or 0),
            cached_prompt_tokens=cached_prompt_tokens,
            prompt_audio_tokens=prompt_audio_tokens,
            completion_audio_tokens=completion_audio_tokens,
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


def _chat_kwargs(
    request: LLMRequest,
    *,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = request.temperature
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    return kwargs


def _responses_kwargs(
    request: LLMRequest,
    *,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": request.model,
        "input": [{"role": "user", "content": request.prompt}],
    }
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    return kwargs


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

    def _invoke_chat_completions(self, client: Any, request: LLMRequest):
        return client.chat.completions.create(
            **_chat_kwargs(
                request,
                reasoning_effort=self._reasoning_effort,
                max_output_tokens=self._max_output_tokens,
            )
        )

    def _invoke_responses_api(self, client: Any, request: LLMRequest):
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
