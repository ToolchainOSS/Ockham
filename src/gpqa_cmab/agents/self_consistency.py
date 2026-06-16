from __future__ import annotations

import random
from collections import Counter

from gpqa_cmab.dataset import question_context
from gpqa_cmab.json_utils import build_record_kwargs, complete_validated
from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.prompts import load_prompt, prompt_version
from gpqa_cmab.schemas import GPQAQuestion, LLMRequest, SelfConsistencyOutput
from gpqa_cmab.telemetry import TelemetryLogger


def run_self_consistency(
    client: LLMClient,
    question: GPQAQuestion,
    *,
    k: int,
    seed: int,
    experiment_id: str,
    model: str,
    telemetry: TelemetryLogger,
    temperature: float = 0.7,
) -> SelfConsistencyOutput:
    answers: list[SelfConsistencyOutput] = []
    prompt_name = "self_consistency_v1"
    for sample in range(k):
        prompt = (
            f"{load_prompt(prompt_name)}\n\n{question_context(question)}\n\n"
            f"Sample index: {sample}"
        )
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            metadata={
                "agent_type": "self_consistency",
                "mock_correct_answer": question.correct_answer,
            },
        )
        record_kwargs = build_record_kwargs(
            experiment_id=experiment_id,
            question_id=question.question_id,
            agent_type="self_consistency",
            subset_id=f"SC-{k}",
            model=model,
            prompt_version=prompt_version(prompt_name),
            temperature=temperature,
        )
        parsed, _ = complete_validated(
            client,
            request,
            SelfConsistencyOutput,
            telemetry=telemetry,
            record_kwargs=record_kwargs,
        )
        answers.append(parsed)
    counts = Counter(item.final_answer for item in answers)
    max_count = max(counts.values())
    tied = [answer for answer, count in counts.items() if count == max_count]
    final = random.Random(seed).choice(sorted(tied))
    return SelfConsistencyOutput(
        final_answer=final,
        confidence=max_count / k,
        rationale_summary=f"Plurality vote across {k} samples.",
    )
