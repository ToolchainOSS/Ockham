"""Tests for the offline CMAB benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpqa_cmab.experiments.cmab_benchmark import (
    BenchmarkReport,
    SubsetStats,
    default_policy_factories,
    load_subset_stats,
    mvp_subset_stats,
    report_to_jsonable,
    run_benchmark,
)
from gpqa_cmab.experiments.mvp_aggregates import MVP_SUBSET_AGGREGATES


def test_mvp_aggregates_table_has_all_16_subsets():
    sids = {sid for sid, _, _ in MVP_SUBSET_AGGREGATES}
    assert len(sids) == 16
    assert "main_only" in sids
    assert "A,B,C,D" in sids


def test_mvp_subset_stats_returns_subset_stats():
    stats = mvp_subset_stats()
    assert all(isinstance(s, SubsetStats) for s in stats)
    by_sid = {s.subset_id: s for s in stats}
    # Spot-check the high-utility subset.
    assert by_sid["A,C"].accuracy == pytest.approx(0.849)
    assert by_sid["A,C"].size == 2


def test_load_subset_stats_round_trips(tmp_path: Path):
    payload = {
        "subsets": [
            {"subset_id": "main_only", "accuracy": 0.4, "avg_tokens": 100.0},
            {"subset_id": "A,C", "accuracy": 0.9, "avg_tokens": 500.0},
        ]
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    stats = load_subset_stats(path)
    by_sid = {s.subset_id: s for s in stats}
    assert by_sid["main_only"].size == 0
    assert by_sid["A,C"].size == 2


def test_run_benchmark_rejects_smoke_summary(tmp_path: Path):
    # Every subset at accuracy 1.0 looks like a single-question mock smoke.
    payload = {
        "subsets": [
            {"subset_id": sid, "accuracy": 1.0, "avg_tokens": 100.0}
            for sid, *_ in MVP_SUBSET_AGGREGATES
        ]
    }
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="single-question smoke"):
        run_benchmark(path, n_seeds=2, n_steps=4)


def test_run_benchmark_fixed_cmab_beats_legacy_on_canonical_data():
    """Smoke regression test for the cold-start bug fix.

    On the canonical 86-question aggregates the *legacy* StructuredCMAB
    collapses to ``main_only`` and earns utility ~0.50; the fixed
    version explores and earns ~0.65. Use a small seed/step budget so
    the test stays fast.
    """
    report = run_benchmark(None, n_seeds=30, n_steps=86)
    assert isinstance(report, BenchmarkReport)
    by_name = {p.name: p for p in report.policies}
    fixed = by_name["structured-cmab (fixed)"]
    legacy = by_name["structured-cmab (legacy-buggy)"]
    # The fix must move the policy off the main_only basin.
    assert fixed.utility_mean > legacy.utility_mean
    # And it must visit many more unique subsets in the limit.
    assert fixed.unique_subsets_mean > legacy.unique_subsets_mean


def test_default_policy_factories_includes_both_versions():
    fac = default_policy_factories()
    assert "structured-cmab (fixed)" in fac
    assert "structured-cmab (legacy-buggy)" in fac
    assert "superarm-ts (fixed)" in fac
    assert "superarm-ts (legacy-flat-prior)" in fac


def test_report_to_jsonable_is_serialisable():
    report = run_benchmark(None, n_seeds=3, n_steps=8)
    payload = report_to_jsonable(report)
    assert json.dumps(payload)  # must not raise
