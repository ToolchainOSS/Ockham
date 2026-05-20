from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from gpqa_cmab.schemas import LLMRequest

ModelT = TypeVar("ModelT", bound=BaseModel)


def parse_json_with_retries(
    invoke: Callable[[str], str],
    request: LLMRequest,
    model_type: type[ModelT],
    max_retries: int = 2,
) -> ModelT:
    prompt = request.prompt
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        content = invoke(prompt)
        try:
            return model_type.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            prompt = (
                f"{request.prompt}\n\n"
                f"Previous JSON was invalid on attempt {attempt + 1}: "
                f"{exc}. Return only valid JSON."
            )
    raise ValueError(f"Failed to parse JSON after retries: {last_error}")
