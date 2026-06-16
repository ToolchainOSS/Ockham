"""Request-kwarg construction and response normalization for both the Chat
Completions and Responses APIs. Vendor-neutral: operates on duck-typed
SDK response objects without importing the ``openai`` package.
"""

from __future__ import annotations

from typing import Any

from gpqa_cmab.schemas import LLMRequest, LLMResponse, Usage


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
