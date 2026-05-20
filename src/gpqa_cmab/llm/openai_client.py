from __future__ import annotations

import os
import time

from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.schemas import LLMRequest, LLMResponse, Usage


class OpenAIClient(LLMClient):
    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install gpqa-cmab[openai] to use OpenAIClient") from exc
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=request.model,
            temperature=request.temperature,
            messages=[{"role": "user", "content": request.prompt}],
        )
        usage = response.usage
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
