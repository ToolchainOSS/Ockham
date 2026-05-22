from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

Answer = Literal["A", "B", "C", "D"]
AnswerOrNull = Answer | None


class Usage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated: bool = False
    # Reasoning tokens are billed inside `completion_tokens` by OpenAI but are
    # surfaced separately here for telemetry. Defaults to 0 for non-reasoning
    # responses so existing callers and fixtures keep working unchanged.
    reasoning_tokens: int = Field(default=0, ge=0)

    @field_validator("total_tokens")
    @classmethod
    def total_is_consistent(cls, value: int, info: Any) -> int:
        prompt = info.data.get("prompt_tokens", 0)
        completion = info.data.get("completion_tokens", 0)
        if value < prompt + completion:
            raise ValueError("total_tokens must be at least prompt + completion")
        return value


class LLMRequest(BaseModel):
    prompt: str
    model: str
    temperature: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str
    usage: Usage
    latency_ms: int = Field(ge=0)
    raw_response: Any | None = None


class GPQAQuestion(BaseModel):
    question_id: str
    domain: str
    question: str
    choices: dict[Answer, str]
    correct_answer: Answer

    @field_validator("domain")
    @classmethod
    def lowercase_domain(cls, value: str) -> str:
        return value.strip().lower()


class ReferenceFact(BaseModel):
    fact: str
    relevance: Literal["high", "medium", "low"]
    source_type: Literal["model_memory", "retrieved_source"]
    source: str | None = None


class SubagentAReport(BaseModel):
    subagent: Literal["physics_specialist"]
    core_principles: list[str]
    reasoning_summary: str
    option_analysis: dict[Answer, str]
    recommended_answer: Answer
    confidence: float = Field(ge=0.0, le=1.0)
    known_uncertainties: list[str]


class SubagentBReport(BaseModel):
    subagent: Literal["reference_retrieval"]
    relevant_facts: list[ReferenceFact]
    candidate_equations: list[str]
    candidate_constants: list[str]
    retrieval_caveats: list[str]
    recommended_answer_if_any: AnswerOrNull
    confidence: float = Field(ge=0.0, le=1.0)


class ComputedResult(BaseModel):
    quantity: str
    value: str
    unit: str


class SubagentCReport(BaseModel):
    subagent: Literal["computational_checker"]
    calculation_needed: bool
    calculation_type: Literal[
        "unit_conversion",
        "algebra",
        "dimensional_analysis",
        "order_of_magnitude",
        "other",
        "none",
    ]
    assumptions: list[str]
    work_summary: str
    computed_results: list[ComputedResult]
    option_consistency: dict[Answer, Literal["consistent", "inconsistent", "unknown"]]
    recommended_answer: AnswerOrNull
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: list[str]


class OptionAudit(BaseModel):
    status: Literal["supported", "rejected", "uncertain"]
    reason: str


class SubagentDReport(BaseModel):
    subagent: Literal["adversarial_verifier"]
    option_audit: dict[Answer, OptionAudit]
    detected_failure_modes: list[str]
    surviving_options: list[Answer]
    recommended_answer: AnswerOrNull
    confidence: float = Field(ge=0.0, le=1.0)


SubagentReport = SubagentAReport | SubagentBReport | SubagentCReport | SubagentDReport


class MainIntegratorOutput(BaseModel):
    final_answer: Answer
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_summary: str
    subagent_influence: dict[
        Answer, Literal["positive", "negative", "neutral", "not_used"]
    ]


class SelfConsistencyOutput(BaseModel):
    final_answer: Answer
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_summary: str


class CallTelemetry(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    experiment_id: str
    question_id: str
    agent_type: Literal["main", "A", "B", "C", "D", "self_consistency"]
    subset_id: str
    model: str
    prompt_version: str
    temperature: float
    attempt: int = Field(default=1, ge=1)
    prompt_text: str | None = None
    prompt_sha256: str | None = None
    prompt_chars: int | None = Field(default=None, ge=0)
    response_text: str | None = None
    response_sha256: str | None = None
    response_chars: int | None = Field(default=None, ge=0)
    raw_response: Any | None = None
    raw_response_sha256: str | None = None
    schema_name: str | None = None
    request_metadata: dict[str, str] = Field(default_factory=dict)
    request_metadata_keys: list[str] = Field(default_factory=list)
    usage: Usage
    latency_ms: int
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    success: bool
    error_type: str | None = None
    error_message: str | None = None


class AggregateTelemetry(BaseModel):
    experiment_id: str
    question_id: str
    subset_id: str
    selected_subagents: list[str]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    main_tokens: int
    subagent_tokens: dict[str, int]
    num_subagent_calls: int
    latency_total_ms: int
    estimated_cost_usd: float = 0.0


class FactorialResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    question_id: str
    domain: str
    subset_id: str
    selected_subagents: list[str]
    final_answer: Answer
    correct_answer: Answer
    correct: bool
    confidence: float
    usage: AggregateTelemetry
    prompt_versions: dict[str, str]


class BanditStep(BaseModel):
    seed: int
    step: int
    policy: str
    question_id: str
    selected_subset_id: str
    correct: bool
    total_tokens: int
    utility: float
    cumulative_utility: float
    unique_subsets_explored: int
