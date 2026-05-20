from __future__ import annotations

from gpqa_cmab.json_utils import parse_json_with_retries
from gpqa_cmab.schemas import LLMRequest, MainIntegratorOutput


def test_json_parse_retry_behavior():
    calls = iter(
        [
            "not json",
            '{"final_answer":"A","confidence":0.8,"rationale_summary":"ok","subagent_influence":{"A":"not_used","B":"not_used","C":"not_used","D":"not_used"}}',
        ]
    )
    parsed = parse_json_with_retries(
        lambda _: next(calls), LLMRequest(prompt="p", model="m"), MainIntegratorOutput
    )
    assert parsed.final_answer == "A"
