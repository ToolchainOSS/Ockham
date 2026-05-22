from gpqa_cmab.cost_guard import (
    CostGuard,
    CostRates,
    estimate_usage_cost_usd,
    usage_cost_breakdown,
)
from gpqa_cmab.schemas import Usage


def test_tiered_cost_estimate_uses_cached_input_discount_and_output_rate():
    usage = Usage(
        prompt_tokens=2006,
        cached_prompt_tokens=1920,
        completion_tokens=300,
        total_tokens=2306,
        reasoning_tokens=40,
    )
    rates = CostRates(
        input_usd_per_1m_tokens=2.50,
        cached_input_usd_per_1m_tokens=0.25,
        output_usd_per_1m_tokens=15.00,
    )

    cost = estimate_usage_cost_usd(usage, rates)

    assert cost == (86 * 2.50 + 1920 * 0.25 + 300 * 15.00) / 1_000_000


def test_cost_guard_enforces_tiered_usd_cap_from_usage():
    guard = CostGuard(
        max_estimated_cost_usd=0.001,
        cost_input_usd_per_1m_tokens=1.0,
        cost_cached_input_usd_per_1m_tokens=0.1,
        cost_output_usd_per_1m_tokens=10.0,
    )
    usage = Usage(prompt_tokens=100, completion_tokens=100, total_tokens=200)

    guard.add_call_usage(usage)

    assert guard.calls == 1
    assert guard.total_tokens == 200
    assert guard.estimated_cost_usd == 0.0011
    assert guard.exhausted()


def test_missing_tiered_rates_are_filled_from_max_configured_rate():
    rates = CostRates(input_usd_per_1m_tokens=1.0, output_usd_per_1m_tokens=10.0)
    usage = Usage(
        prompt_tokens=10, cached_prompt_tokens=5, completion_tokens=10, total_tokens=20
    )

    assert rates.input_usd_per_1m_tokens == 1.0
    assert rates.cached_input_usd_per_1m_tokens == 10.0
    assert rates.output_usd_per_1m_tokens == 10.0
    assert (
        estimate_usage_cost_usd(usage, rates)
        == (5 * 1.0 + 5 * 10.0 + 10 * 10.0) / 1_000_000
    )


def test_usage_cost_breakdown_preserves_audio_counts_but_costs_text_tokens():
    usage = Usage(
        prompt_tokens=100,
        cached_prompt_tokens=30,
        prompt_audio_tokens=10,
        completion_tokens=20,
        completion_audio_tokens=5,
        total_tokens=120,
    )
    rates = CostRates(
        input_usd_per_1m_tokens=1.0,
        cached_input_usd_per_1m_tokens=0.5,
        output_usd_per_1m_tokens=2.0,
    )

    breakdown = usage_cost_breakdown([usage], rates)

    assert breakdown["uncached_input_tokens"] == 60
    assert breakdown["cached_input_tokens"] == 30
    assert breakdown["output_tokens"] == 15
    assert breakdown["prompt_audio_tokens"] == 10
    assert breakdown["completion_audio_tokens"] == 5
    assert (
        breakdown["estimated_cost_usd"] == (60 * 1.0 + 30 * 0.5 + 15 * 2.0) / 1_000_000
    )
