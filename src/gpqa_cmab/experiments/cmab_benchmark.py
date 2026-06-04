"""Offline CMAB benchmark on the empirical GPQA-Diamond utility surface.

Why this exists
---------------
The original ``run-factorial`` artifact lives outside git (it costs real
LLM tokens to produce). To answer the *bandit-algorithm* question
"after fixing the cold-start bugs, does StructuredCMAB still lose to
sub-agent C alone?" we need many seeds against a known reward surface.

This module builds a **synthetic bandit environment** from the
per-subset aggregates in ``artifacts/results/metrics_summary.json``:

* The reward for ``select(subset)`` is a Bernoulli draw from the
  observed accuracy of that subset on the real 86-question MVP.
* The cost is the observed ``avg_tokens`` of that subset.
* The score the bandit optimizes is the utility used everywhere else in
  the project: ``1{correct} - λ_token · tokens/avg_all_four - λ_call · |S|``.

The environment therefore preserves the real accuracy / cost trade-off
of the dataset while letting us run as many replays as we want without
spending a token. The CMAB algorithms are unchanged — we just hand them
deterministic per-question outcomes drawn from the per-subset accuracy
distributions.

This is a strict superset of what the original ``replay_bandit``
function does for one factorial file: it can run many algorithms over
many seeds and emit a single comparable summary.
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gpqa_cmab.bandits.structured_cmab import StructuredCMAB
from gpqa_cmab.bandits.superarm_ts import SuperArmThompsonSampler
from gpqa_cmab.experiments.mvp_aggregates import MVP_SUBSET_AGGREGATES
from gpqa_cmab.subsets import all_subsets, subset_id

# Policy factory signature: ``(seed, lambda_token, lambda_call) -> policy``.
PolicyFactory = Callable[[int, float, float], object]


def _utility(
    correct: bool,
    tokens: float,
    size: int,
    avg_all4: float,
    lambda_token: float,
    lambda_call: float,
) -> float:
    normalized = tokens / avg_all4 if avg_all4 else 0.0
    return float(correct) - lambda_token * normalized - lambda_call * size


@dataclass
class SubsetStats:
    """Aggregate accuracy + cost for one subset."""

    subset_id: str
    accuracy: float
    avg_tokens: float
    size: int


def load_subset_stats(metrics_summary_path: Path) -> list[SubsetStats]:
    """Read ``metrics_summary.json`` and return per-subset accuracy/cost."""
    raw = json.loads(metrics_summary_path.read_text(encoding="utf-8"))
    out: list[SubsetStats] = []
    for row in raw["subsets"]:
        sid = row["subset_id"]
        size = 0 if sid == "main_only" else len(sid.split(","))
        out.append(
            SubsetStats(
                subset_id=sid,
                accuracy=float(row["accuracy"]),
                avg_tokens=float(row["avg_tokens"]),
                size=size,
            )
        )
    return out


def mvp_subset_stats() -> list[SubsetStats]:
    """Canonical 86-question MVP aggregates, baked into the source tree.

    Used as the default benchmark environment so the offline comparison
    stays reproducible even when ``metrics_summary.json`` has been
    overwritten by a downstream smoke-test or quick-check run.
    """
    return [
        SubsetStats(
            subset_id=sid,
            accuracy=acc,
            avg_tokens=tokens,
            size=0 if sid == "main_only" else len(sid.split(",")),
        )
        for sid, acc, tokens in MVP_SUBSET_AGGREGATES
    ]


@dataclass
class PolicyResult:
    """Single-policy summary across all seeds."""

    name: str
    n_seeds: int
    n_steps: int
    accuracy_mean: float
    accuracy_std: float
    tokens_mean: float
    tokens_std: float
    utility_mean: float  # mean cumulative utility per seed, /n_steps
    utility_std: float
    final_cum_utility_mean: float
    unique_subsets_mean: float
    pick_distribution: dict[str, float]  # over all (seed, step) picks
    late_window_distribution: dict[str, float]  # last 30 steps per seed


@dataclass
class BenchmarkReport:
    """Top-level comparison across policies + static baselines."""

    n_seeds: int
    n_steps: int
    lambda_token: float
    lambda_call: float
    avg_all_four_tokens: float
    static_baselines: dict[str, dict[str, float]]
    policies: list[PolicyResult] = field(default_factory=list)


def _run_static_baselines(
    stats: list[SubsetStats],
    n_seeds: int,
    n_steps: int,
    lambda_token: float,
    lambda_call: float,
    avg_all4: float,
    rng_seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Simulate fixed-subset baselines under the same Bernoulli environment.

    Each baseline plays the same subset every step; we still draw per-step
    Bernoulli outcomes so the noise floor matches the bandit comparison.
    """
    by_sid = {s.subset_id: s for s in stats}
    targets = ["main_only", "A", "C", "A,C", "A,B,C", "A,B,C,D"]
    out: dict[str, dict[str, float]] = {}
    for sid in targets:
        if sid not in by_sid:
            continue
        s = by_sid[sid]
        utilities: list[float] = []
        accuracies: list[float] = []
        for seed in range(n_seeds):
            rng = random.Random(rng_seed + seed)
            seed_utils: list[float] = []
            seed_acc: list[float] = []
            for _ in range(n_steps):
                correct = rng.random() < s.accuracy
                seed_utils.append(
                    _utility(
                        correct,
                        s.avg_tokens,
                        s.size,
                        avg_all4,
                        lambda_token,
                        lambda_call,
                    )
                )
                seed_acc.append(float(correct))
            utilities.append(statistics.mean(seed_utils))
            accuracies.append(statistics.mean(seed_acc))
        out[sid] = {
            "accuracy_mean": statistics.mean(accuracies),
            "avg_tokens": s.avg_tokens,
            "utility_mean": statistics.mean(utilities),
            "utility_std": statistics.stdev(utilities) if n_seeds > 1 else 0.0,
            "size": s.size,
        }
    return out


def _run_policy(
    name: str,
    factory: PolicyFactory,
    stats: list[SubsetStats],
    n_seeds: int,
    n_steps: int,
    lambda_token: float,
    lambda_call: float,
    avg_all4: float,
    late_window: int = 30,
    rng_seed: int = 0,
) -> PolicyResult:
    by_sid = {s.subset_id: s for s in stats}
    token_costs = {s.subset_id: s.avg_tokens for s in stats}
    all_sids = [subset_id(sub) for sub in all_subsets()]
    accuracies: list[float] = []
    tokens: list[float] = []
    utilities: list[float] = []
    final_cums: list[float] = []
    uniques: list[int] = []
    all_picks: list[str] = []
    late_picks: list[str] = []

    for seed in range(n_seeds):
        env_rng = random.Random(rng_seed + seed)
        policy = factory(seed, lambda_token, lambda_call)
        seen: set[str] = set()
        seed_acc: list[float] = []
        seed_tokens: list[float] = []
        seed_util: list[float] = []
        for step in range(n_steps):
            pick = policy.select(token_costs, avg_all4)
            stat = by_sid[pick]
            correct = env_rng.random() < stat.accuracy
            u = _utility(
                correct,
                stat.avg_tokens,
                stat.size,
                avg_all4,
                lambda_token,
                lambda_call,
            )
            seed_acc.append(float(correct))
            seed_tokens.append(stat.avg_tokens)
            seed_util.append(u)
            seen.add(pick)
            all_picks.append(pick)
            if step >= n_steps - late_window:
                late_picks.append(pick)
            # Both bandit classes expose a ``.update`` method but with
            # different signatures; dispatch on attribute introspection.
            if isinstance(policy, StructuredCMAB):
                policy.update(pick, correct, stat.avg_tokens, avg_all4)
            else:
                policy.update(pick, correct)
        accuracies.append(statistics.mean(seed_acc))
        tokens.append(statistics.mean(seed_tokens))
        utilities.append(statistics.mean(seed_util))
        final_cums.append(sum(seed_util))
        uniques.append(len(seen))

    def _hist(picks: list[str]) -> dict[str, float]:
        total = len(picks) or 1
        return {
            sid: round(picks.count(sid) / total, 4)
            for sid in all_sids
            if picks.count(sid)
        }

    return PolicyResult(
        name=name,
        n_seeds=n_seeds,
        n_steps=n_steps,
        accuracy_mean=statistics.mean(accuracies),
        accuracy_std=statistics.stdev(accuracies) if n_seeds > 1 else 0.0,
        tokens_mean=statistics.mean(tokens),
        tokens_std=statistics.stdev(tokens) if n_seeds > 1 else 0.0,
        utility_mean=statistics.mean(utilities),
        utility_std=statistics.stdev(utilities) if n_seeds > 1 else 0.0,
        final_cum_utility_mean=statistics.mean(final_cums),
        unique_subsets_mean=statistics.mean(uniques),
        pick_distribution=_hist(all_picks),
        late_window_distribution=_hist(late_picks),
    )


def default_policy_factories() -> dict[str, PolicyFactory]:
    """Return the policies compared by the benchmark.

    We include both the fixed (default) and the legacy (buggy) versions
    so the offline report can show the effect of the bug fixes
    side-by-side.
    """

    def fixed_structured(seed: int, lt: float, lc: float) -> object:
        return StructuredCMAB(seed=seed, lambda_token=lt, lambda_call=lc)

    def legacy_structured(seed: int, lt: float, lc: float) -> object:
        return StructuredCMAB(
            seed=seed,
            lambda_token=lt,
            lambda_call=lc,
            prior_accuracy=0.5,
            shrink_intercept=True,
            uncertainty=0.1,
            bonus_form="inv_sqrt_n",
        )

    def fixed_superarm(seed: int, lt: float, lc: float) -> object:
        return SuperArmThompsonSampler(seed=seed, lambda_token=lt, lambda_call=lc)

    def legacy_superarm(seed: int, lt: float, lc: float) -> object:
        return SuperArmThompsonSampler(
            seed=seed,
            lambda_token=lt,
            lambda_call=lc,
            alpha0=1.0,
            beta0=1.0,
        )

    return {
        "structured-cmab (fixed)": fixed_structured,
        "structured-cmab (legacy-buggy)": legacy_structured,
        "superarm-ts (fixed)": fixed_superarm,
        "superarm-ts (legacy-flat-prior)": legacy_superarm,
    }


def run_benchmark(
    metrics_summary_path: Path | None = None,
    *,
    n_seeds: int = 200,
    n_steps: int = 86,
    lambda_token: float = 0.05,
    lambda_call: float = 0.01,
    rng_seed: int = 0,
    policies: dict[str, PolicyFactory] | None = None,
    stats: list[SubsetStats] | None = None,
) -> BenchmarkReport:
    if stats is None:
        stats = (
            load_subset_stats(metrics_summary_path)
            if metrics_summary_path is not None
            else mvp_subset_stats()
        )
    # Guard against silently benchmarking on a 1-question smoke artifact.
    if all(s.accuracy in (0.0, 1.0) for s in stats):
        raise ValueError(
            "All subsets report accuracy in {0, 1} — the metrics_summary "
            "file looks like a single-question smoke output. Drop the "
            "``--metrics-summary`` flag to use the canonical 86Q MVP "
            "aggregates baked into mvp_aggregates.py."
        )
    avg_all4 = next(s.avg_tokens for s in stats if s.subset_id == "A,B,C,D")
    static = _run_static_baselines(
        stats,
        n_seeds,
        n_steps,
        lambda_token,
        lambda_call,
        avg_all4,
        rng_seed,
    )
    policy_factories = policies if policies is not None else default_policy_factories()
    policy_results = [
        _run_policy(
            name,
            factory,
            stats,
            n_seeds,
            n_steps,
            lambda_token,
            lambda_call,
            avg_all4,
            rng_seed=rng_seed,
        )
        for name, factory in policy_factories.items()
    ]
    return BenchmarkReport(
        n_seeds=n_seeds,
        n_steps=n_steps,
        lambda_token=lambda_token,
        lambda_call=lambda_call,
        avg_all_four_tokens=avg_all4,
        static_baselines=static,
        policies=policy_results,
    )


def report_to_jsonable(report: BenchmarkReport) -> dict[str, object]:
    """Convert ``BenchmarkReport`` to a plain dict for ``json.dump``."""
    out = asdict(report)
    out["policies"] = [asdict(p) for p in report.policies]
    return out
