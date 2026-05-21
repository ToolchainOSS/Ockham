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

logger = logging.getLogger(__name__)


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
    ``cost_usd_per_1k_tokens`` is the assumed flat rate used to translate
    tokens into USD for the cost cap. When it is ``0.0`` (the default) the
    USD cap is treated as inactive even if set — and a warning is logged
    once so users notice their cap is silently inert.
    """

    max_api_calls: int | None = None
    max_estimated_cost_usd: float | None = None
    cost_usd_per_1k_tokens: float = 0.0
    calls: int = 0
    total_tokens: int = 0
    _warned_zero_rate: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self.max_estimated_cost_usd is not None
            and self.cost_usd_per_1k_tokens <= 0.0
            and not self._warned_zero_rate
        ):
            logger.warning(
                "cost_guard_zero_rate cap=%.4f USD set but "
                "cost_usd_per_1k_tokens=0; USD cap is INACTIVE. Set "
                "COST_USD_PER_1K_TOKENS to enable.",
                self.max_estimated_cost_usd,
            )
            self._warned_zero_rate = True

    @property
    def estimated_cost_usd(self) -> float:
        return self.total_tokens / 1000.0 * self.cost_usd_per_1k_tokens

    def add_call(self, tokens: int) -> None:
        """Record one completed API call's token usage."""
        self.calls += 1
        self.total_tokens += max(0, int(tokens))

    def exhausted(self) -> bool:
        if self.max_api_calls is not None and self.calls >= self.max_api_calls:
            return True
        return (
            self.max_estimated_cost_usd is not None
            and self.cost_usd_per_1k_tokens > 0.0
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
        return {
            "calls": self.calls,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "max_api_calls": self.max_api_calls,
            "max_estimated_cost_usd": self.max_estimated_cost_usd,
            "cost_usd_per_1k_tokens": self.cost_usd_per_1k_tokens,
        }
