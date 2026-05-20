from __future__ import annotations

from abc import ABC, abstractmethod

from gpqa_cmab.schemas import LLMRequest, LLMResponse


class LLMClient(ABC):
    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError
