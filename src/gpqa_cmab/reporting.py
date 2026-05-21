from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from gpqa_cmab.metrics import baseline_summary, subset_table
from gpqa_cmab.schemas import FactorialResult
from gpqa_cmab.telemetry import read_jsonl


def write_evaluation_outputs(rows: list[FactorialResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = subset_table(rows)
    with (output_dir / "subset_accuracy_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(table[0]) if table else ["subset_id"]
        )
        writer.writeheader()
        writer.writerows(table)
    summary = {
        "num_questions": len({row.question_id for row in rows}),
        "num_rows": len(rows),
        "subsets": table,
        "baselines": baseline_summary(rows),
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def _bandit_replay_summary(replay_path: Path) -> dict[str, object]:
    """Aggregate bandit replay rows: oracle regret, unique subset coverage."""
    if not replay_path.exists():
        return {}
    rows = read_jsonl(replay_path)
    if not rows:
        return {}
    by_policy_seed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_policy_seed[(row["policy"], row["seed"])].append(row)
    selected_subsets: set[str] = {row["selected_subset_id"] for row in rows}
    policies: dict[str, dict[str, float]] = {}
    for (policy, _seed), steps in by_policy_seed.items():
        steps.sort(key=lambda entry: entry["step"])
        final = steps[-1]
        policies.setdefault(
            policy,
            {
                "final_cumulative_utility": [],
                "final_unique_subsets_explored": [],
                "avg_tokens": [],
                "accuracy": [],
            },
        )
        policies[policy]["final_cumulative_utility"].append(final["cumulative_utility"])
        policies[policy]["final_unique_subsets_explored"].append(
            final["unique_subsets_explored"]
        )
        policies[policy]["avg_tokens"].append(
            mean(step["total_tokens"] for step in steps)
        )
        policies[policy]["accuracy"].append(
            mean(float(step["correct"]) for step in steps)
        )
    aggregated = {
        policy: {key: mean(values) for key, values in metrics.items()}
        for policy, metrics in policies.items()
    }
    return {
        "policies": aggregated,
        "unique_subsets_explored_total": len(selected_subsets),
        "subset_space_size": 16,
        "fraction_subset_space_explored": len(selected_subsets) / 16,
    }


def write_report(results_dir: Path, output: Path) -> None:
    summary_path = results_dir / "metrics_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    subsets = summary.get("subsets", [])
    baselines = summary.get("baselines", {})
    all_four = next((row for row in subsets if row["subset_id"] == "A,B,C,D"), None)
    main = next((row for row in subsets if row["subset_id"] == "main_only"), None)
    replay_summary = _bandit_replay_summary(results_dir / "bandit_replay_results.jsonl")
    oracle = baselines.get("oracle_fixed_subset", {})
    lines = [
        "# GPQA-Diamond Physics CMAB MVP Report",
        "",
        "This report evaluates cost-aware subagent pruning, not leaderboard accuracy.",
        "",
        "## Dataset Summary",
        f"- Questions: {summary.get('num_questions', 0)}",
        "- Domain: physics",
        "",
        "## Key References",
        f"- Main-only accuracy: {_fmt(main, 'accuracy')}",
        f"- All-four accuracy: {_fmt(all_four, 'accuracy')}",
        f"- All-four average tokens: {_fmt(all_four, 'avg_tokens')}",
        "",
        "## Accuracy By Subset",
    ]
    for row in subsets:
        lines.append(
            f"- {row['subset_id']}: accuracy={row['accuracy']:.3f}, "
            f"avg_tokens={row['avg_tokens']:.1f}, utility={row['utility']:.3f}"
        )
    static = baselines.get("static_pruning", {})
    random_bl = baselines.get("random_budget_matched", {})
    static_sid = static.get("subset_id", "n/a") if isinstance(static, dict) else "n/a"
    random_target = (
        random_bl.get("target_subset_id", "n/a")
        if isinstance(random_bl, dict)
        else "n/a"
    )
    oracle_sid = oracle.get("subset_id", "n/a") if isinstance(oracle, dict) else "n/a"
    lines.extend(
        [
            "",
            "## Baselines",
            (
                f"- Static pruning ({static_sid}): "
                f"accuracy={_fmt(static, 'accuracy')}, "
                f"avg_tokens={_fmt(static, 'avg_tokens')}, "
                f"utility={_fmt(static, 'utility')}"
            ),
            (
                f"- Random budget-matched (target {random_target}): "
                f"accuracy={_fmt(random_bl, 'accuracy')}, "
                f"avg_tokens={_fmt(random_bl, 'avg_tokens')}, "
                f"utility={_fmt(random_bl, 'utility')}"
            ),
            (
                f"- Oracle fixed-subset (MVP only): subset={oracle_sid}, "
                f"utility={_fmt(oracle, 'utility')}, "
                f"accuracy={_fmt(oracle, 'accuracy')}, "
                f"avg_tokens={_fmt(oracle, 'avg_tokens')}"
            ),
            "",
            "## CMAB Replay",
            "Bandit replay observes only the selected subset outcome at each step.",
        ]
    )
    policies = replay_summary.get("policies", {}) if replay_summary else {}
    if policies:
        oracle_utility = (
            float(oracle.get("utility", 0.0))
            if isinstance(oracle, dict) and "utility" in oracle
            else 0.0
        )
        for policy, metrics in policies.items():
            final_util = metrics["final_cumulative_utility"]
            steps_count = max(int(metrics.get("final_unique_subsets_explored", 1)), 1)
            avg_step_utility = final_util / steps_count if steps_count else 0.0
            explored = metrics["final_unique_subsets_explored"]
            lines.append(
                f"- Policy {policy}: avg accuracy={metrics['accuracy']:.3f}, "
                f"avg tokens={metrics['avg_tokens']:.1f}, "
                f"final cumulative utility={final_util:.3f}, "
                f"unique subsets explored={explored:.1f}, "
                f"regret vs oracle={oracle_utility - avg_step_utility:+.3f}"
            )
        lines.append(
            f"- Subset space explored: "
            f"{replay_summary.get('unique_subsets_explored_total', 0)}/"
            f"{replay_summary.get('subset_space_size', 16)} "
            f"({replay_summary.get('fraction_subset_space_explored', 0.0):.2%})"
        )
    else:
        lines.append("- No bandit replay artifacts found.")
    lines.extend(
        [
            "",
            "## Self-Consistency",
            "Run `gpqa-cmab run-self-consistency` to produce baseline samples "
            "(CoT-1, SC-4, SC-8, SC-16). Outputs land in "
            "`artifacts/results/self_consistency_results.jsonl`.",
            "",
            "## Limitations",
            "Mock mode is deterministic and validates orchestration only. "
            "Real claims require live model runs, repeated seeds, and bootstrap "
            "intervals over enough questions. Oracle fixed-subset is an analysis "
            "reference for the small MVP and not a deployable policy.",
            "",
            "## Next Steps",
            "Run live provider experiments with cost caps, add richer uncertainty "
            "reporting, and scale beyond four optional subagents.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(row: dict | None, key: str) -> str:
    if not row or key not in row:
        return "n/a"
    value = row[key]
    return f"{value:.3f}" if isinstance(value, float) else str(value)
