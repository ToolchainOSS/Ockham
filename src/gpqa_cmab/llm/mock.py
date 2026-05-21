from __future__ import annotations

import json
import re
from hashlib import sha256

from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.schemas import LLMRequest, LLMResponse, Usage


class MockLLMClient(LLMClient):
    def complete(self, request: LLMRequest) -> LLMResponse:
        agent = request.metadata.get("agent_type", "main")
        # The mock client returns the correct answer when the upstream
        # prompt-builder passes it via metadata. This keeps the hint out of
        # the actual prompt string so it is never sent to billable LLMs.
        hint = request.metadata.get("mock_correct_answer")
        if hint and hint in "ABCD":
            answer = hint
        else:
            answer = _answer_from_prompt(request.prompt)
        payload = _payload(agent, answer)
        prompt_tokens = max(1, len(request.prompt.split()))
        completion = json.dumps(payload, sort_keys=True)
        completion_tokens = max(1, len(completion.split()))
        return LLMResponse(
            content=completion,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                estimated=True,
            ),
            latency_ms=1,
            raw_response={"mock": True},
        )


def _answer_from_prompt(prompt: str) -> str:
    match = re.search(r"MOCK_CORRECT_ANSWER\s*=\s*([ABCD])", prompt)
    if match:
        return match.group(1)
    return "ABCD"[int(sha256(prompt.encode()).hexdigest(), 16) % 4]


def _payload(agent: str, answer: str) -> dict:
    if agent == "A":
        return {
            "subagent": "physics_specialist",
            "core_principles": ["mock principle"],
            "reasoning_summary": "Mock specialist summary.",
            "option_analysis": {key: "mock analysis" for key in "ABCD"},
            "recommended_answer": answer,
            "confidence": 0.7,
            "known_uncertainties": [],
        }
    if agent == "B":
        return {
            "subagent": "reference_retrieval",
            "relevant_facts": [
                {
                    "fact": "mock fact",
                    "relevance": "medium",
                    "source_type": "model_memory",
                    "source": None,
                }
            ],
            "candidate_equations": [],
            "candidate_constants": [],
            "retrieval_caveats": [],
            "recommended_answer_if_any": answer,
            "confidence": 0.6,
        }
    if agent == "C":
        return {
            "subagent": "computational_checker",
            "calculation_needed": False,
            "calculation_type": "none",
            "assumptions": [],
            "work_summary": "No computation in mock mode.",
            "computed_results": [],
            "option_consistency": {key: "unknown" for key in "ABCD"},
            "recommended_answer": answer,
            "confidence": 0.55,
            "caveats": [],
        }
    if agent == "D":
        return {
            "subagent": "adversarial_verifier",
            "option_audit": {
                key: {
                    "status": "supported" if key == answer else "uncertain",
                    "reason": "mock",
                }
                for key in "ABCD"
            },
            "detected_failure_modes": [],
            "surviving_options": [answer],
            "recommended_answer": answer,
            "confidence": 0.65,
        }
    return {
        "final_answer": answer,
        "confidence": 0.75,
        "rationale_summary": "Mock compact rationale.",
        "subagent_influence": {key: "not_used" for key in "ABCD"},
    }
