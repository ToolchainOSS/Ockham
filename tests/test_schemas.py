from __future__ import annotations

import pytest
from pydantic import ValidationError

from gpqa_cmab.schemas import GPQAQuestion, MainIntegratorOutput, Usage


def test_schema_validation_rejects_bad_answer():
    with pytest.raises(ValidationError):
        GPQAQuestion(
            question_id="q",
            domain="physics",
            question="q",
            choices={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="E",
        )


def test_usage_total_must_be_consistent():
    with pytest.raises(ValidationError):
        Usage(prompt_tokens=3, completion_tokens=3, total_tokens=5)


def test_main_integrator_schema():
    parsed = MainIntegratorOutput.model_validate(
        {
            "final_answer": "A",
            "confidence": 0.5,
            "rationale_summary": "brief",
            "subagent_influence": {
                "A": "not_used",
                "B": "not_used",
                "C": "not_used",
                "D": "not_used",
            },
        }
    )
    assert parsed.final_answer == "A"
