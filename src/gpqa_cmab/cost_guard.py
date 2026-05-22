"""Run-wide cost / call budget enforcement.

The CMAB pipeline issues many LLM calls per question (4 subagents + 16 main
integrator subsets in the full factorial sweep). A single mis-configured run
against a real provider can therefore burn a non-trivial bill. This module
provides a small, sharable ``CostGuard`` that every experiment loop consults
between calls to short-circuit early when a soft budget is exhausted.

The guard is intentionally a *post-call* check: we always allow the call we
already started (telemetry stays consistent), but we stop initiating new ones
once the budget is crossed. Callers that need pre-flight planning (e.g. the
factorial loop that wants whole-question atomicity) can additionally call
``would_exceed_calls(n)`` before starting a question's batch of calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from gpqa_cmab.schemas import Usage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostRates:
    """Provider pricing rates used for local USD estimates.

    Rates are expressed in USD per 1M tokens, matching current OpenAI pricing
    tables. If only one or two rates are configured, the missing rates are
    conservatively filled with the maximum configured rate.
    """

    input_usd_per_1m_tokens: float = 0.0
    cached_input_usd_per_1m_tokens: float = 0.0
    output_usd_per_1m_tokens: float = 0.0

    def __post_init__(self) -> None:
        configured = [
            rate
            for rate in (
                self.input_usd_per_1m_tokens,
                self.cached_input_usd_per_1m_tokens,
                self.output_usd_per_1m_tokens,
            )
            if rate > 0.0
        ]
        if not configured:
            return
        fallback = max(configured)
        if self.input_usd_per_1m_tokens <= 0.0:
            object.__setattr__(self, "input_usd_per_1m_tokens", fallback)
        if self.cached_input_usd_per_1m_tokens <= 0.0:
            object.__setattr__(self, "cached_input_usd_per_1m_tokens", fallback)
        if self.output_usd_per_1m_tokens <= 0.0:
            object.__setattr__(self, "output_usd_per_1m_tokens", fallback)

    @property
    def enabled(self) -> bool:
        return any(
            rate > 0.0
            for rate in (
                self.input_usd_per_1m_tokens,
                self.cached_input_usd_per_1m_tokens,
                self.output_usd_per_1m_tokens,
            )
        )

    def asdict(self) -> dict[str, float | str]:
        return {
            "mode": "tiered" if self.enabled else "unconfigured",
            "input_usd_per_1m_tokens": self.input_usd_per_1m_tokens,
            "cached_input_usd_per_1m_tokens": self.cached_input_usd_per_1m_tokens,
            "output_usd_per_1m_tokens": self.output_usd_per_1m_tokens,
        }


def usage_token_breakdown(usage: Usage) -> dict[str, int]:
    """Split provider usage into the billing buckets used by text models."""
    cached_input = min(usage.cached_prompt_tokens, usage.prompt_tokens)
    prompt_audio = min(usage.prompt_audio_tokens, usage.prompt_tokens - cached_input)
    output_audio = min(usage.completion_audio_tokens, usage.completion_tokens)
    uncached_input = max(0, usage.prompt_tokens - cached_input - prompt_audio)
    text_output = max(0, usage.completion_tokens - output_audio)
    return {
        "uncached_input_tokens": uncached_input,
        "cached_input_tokens": cached_input,
        "output_tokens": text_output,
        "prompt_audio_tokens": prompt_audio,
        "completion_audio_tokens": output_audio,
        "total_tokens": usage.total_tokens,
    }


def estimate_usage_cost_usd(usage: Usage, rates: CostRates) -> float:
    """Estimate USD for one provider call from its reported usage details."""
    if not rates.enabled:
        return 0.0
    tokens = usage_token_breakdown(usage)
    return (
        tokens["uncached_input_tokens"] * rates.input_usd_per_1m_tokens / 1_000_000.0
        + tokens["cached_input_tokens"]
        * rates.cached_input_usd_per_1m_tokens
        / 1_000_000.0
        + tokens["output_tokens"] * rates.output_usd_per_1m_tokens / 1_000_000.0
    )


def usage_cost_breakdown(usages: list[Usage], rates: CostRates) -> dict[str, object]:
    """Aggregate token billing buckets and estimated USD for trace summaries."""
    totals = {
        "uncached_input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "prompt_audio_tokens": 0,
        "completion_audio_tokens": 0,
        "total_tokens": 0,
    }
    estimated_rows = 0
    for usage in usages:
        if usage.estimated:
            estimated_rows += 1
        for key, value in usage_token_breakdown(usage).items():
            totals[key] += value
    return {
        **totals,
        "estimated_usage_rows": estimated_rows,
        "estimated_cost_usd": sum(
            estimate_usage_cost_usd(usage, rates) for usage in usages
        ),
        "pricing": rates.asdict(),
    }


class BudgetExceeded(Exception):
    """Raised when a runtime budget has been crossed.

    Experiment loops should catch this at a safe boundary (e.g. between
    questions) and stop emitting new work without losing already-collected
    rows.
    """


@dataclass
class CostGuard:
    """Track cumulative API calls + tokens and enforce optional caps.

    All caps are optional; an unset (``None``) cap disables that dimension.
    Providers bill input, cached input, and output tokens differently. If only
    one or two rates are configured, missing rates are conservatively filled
    with the maximum configured rate. When no rate is configured, the USD cap
    is inactive and a warning is logged once so users notice their cap is inert.
    """

    max_api_calls: int | None = None
    max_estimated_cost_usd: float | None = None
    cost_input_usd_per_1m_tokens: float = 0.0
    cost_cached_input_usd_per_1m_tokens: float = 0.0
    cost_output_usd_per_1m_tokens: float = 0.0
    calls: int = 0
    total_tokens: int = 0
    _estimated_cost_usd: float = field(default=0.0, repr=False)
    _warned_zero_rate: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self.max_estimated_cost_usd is not None
            and not self.rates.enabled
            and not self._warned_zero_rate
        ):
            logger.warning(
                "cost_guard_zero_rate cap=%.4f USD set but "
                "no pricing rate is configured; USD cap is INACTIVE. Set "
                "COST_INPUT_USD_PER_1M_TOKENS / "
                "COST_CACHED_INPUT_USD_PER_1M_TOKENS / "
                "COST_OUTPUT_USD_PER_1M_TOKENS to enable.",
                self.max_estimated_cost_usd,
            )
            self._warned_zero_rate = True

    @property
    def rates(self) -> CostRates:
        return CostRates(
            input_usd_per_1m_tokens=self.cost_input_usd_per_1m_tokens,
            cached_input_usd_per_1m_tokens=self.cost_cached_input_usd_per_1m_tokens,
            output_usd_per_1m_tokens=self.cost_output_usd_per_1m_tokens,
        )

    @property
    def estimated_cost_usd(self) -> float:
        return self._estimated_cost_usd

    def add_call(self, tokens: int) -> None:
        """Record one completed API call's token usage."""
        self.calls += 1
        self.total_tokens += max(0, int(tokens))

    def add_call_usage(self, usage: Usage) -> None:
        """Record one completed API call using provider billing buckets."""
        self.calls += 1
        self.total_tokens += usage.total_tokens
        self._estimated_cost_usd += estimate_usage_cost_usd(usage, self.rates)

    def exhausted(self) -> bool:
        if self.max_api_calls is not None and self.calls >= self.max_api_calls:
            return True
        return (
            self.max_estimated_cost_usd is not None
            and self.rates.enabled
            and self.estimated_cost_usd >= self.max_estimated_cost_usd
        )

    def would_exceed_calls(self, planned: int) -> bool:
        """True if starting ``planned`` more calls would cross the call cap."""
        if self.max_api_calls is None:
            return False
        return self.calls + max(0, planned) > self.max_api_calls

    def raise_if_exhausted(self) -> None:
        if self.exhausted():
            raise BudgetExceeded(
                f"budget_exhausted calls={self.calls} "
                f"max_calls={self.max_api_calls} "
                f"tokens={self.total_tokens} "
                f"estimated_cost_usd={self.estimated_cost_usd:.4f} "
                f"max_cost_usd={self.max_estimated_cost_usd}"
            )

    def snapshot(self) -> dict[str, float | int | None]:
        rates = self.rates
        return {
            "calls": self.calls,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "max_api_calls": self.max_api_calls,
            "max_estimated_cost_usd": self.max_estimated_cost_usd,
            "cost_input_usd_per_1m_tokens": rates.input_usd_per_1m_tokens,
            "cost_cached_input_usd_per_1m_tokens": rates.cached_input_usd_per_1m_tokens,
            "cost_output_usd_per_1m_tokens": rates.output_usd_per_1m_tokens,
            "pricing_mode": rates.asdict()["mode"],
        }
