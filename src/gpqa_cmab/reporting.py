from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from gpqa_cmab.metrics import (
    baseline_summary,
    cmab_avg_subset_size,
    cmab_avg_tokens,
    non_inferiority_report,
    self_consistency_summary,
    subset_table,
)
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


def _load_replay(replay_path: Path) -> list[dict]:
    if not replay_path.exists():
        return []
    return read_jsonl(replay_path)


def _replay_policy_summary(replay_rows: list[dict]) -> dict[str, dict[str, float]]:
    by_policy_seed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in replay_rows:
        by_policy_seed[(row["policy"], row["seed"])].append(row)
    policies: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "final_cumulative_utility": [],
            "final_unique_subsets_explored": [],
            "avg_tokens": [],
            "accuracy": [],
        }
    )
    for (policy, _seed), steps in by_policy_seed.items():
        steps.sort(key=lambda entry: entry["step"])
        final = steps[-1]
        policies[policy]["final_cumulative_utility"].append(final["cumulative_utility"])
        policies[policy]["final_unique_subsets_explored"].append(
            float(final["unique_subsets_explored"])
        )
        policies[policy]["avg_tokens"].append(
            mean(step["total_tokens"] for step in steps)
        )
        policies[policy]["accuracy"].append(
            mean(float(step["correct"]) for step in steps)
        )
    return {
        policy: {key: mean(values) for key, values in metrics.items()}
        for policy, metrics in policies.items()
    }


def _format_baseline_line(label: str, baseline: dict | None) -> str:
    return (
        f"- {label}: "
        f"accuracy={_fmt(baseline, 'accuracy')}, "
        f"avg_tokens={_fmt(baseline, 'avg_tokens')}, "
        f"utility={_fmt(baseline, 'utility')}"
    )


def write_report(results_dir: Path, output: Path) -> None:
    summary_path = results_dir / "metrics_summary.json"
    factorial_path = results_dir / "full_factorial_results.jsonl"
    replay_path = results_dir / "bandit_replay_results.jsonl"
    sc_path = results_dir / "self_consistency_results.jsonl"

    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    subsets = summary.get("subsets", [])
    baselines = summary.get("baselines", {}) or {}
    all_four = next((row for row in subsets if row["subset_id"] == "A,B,C,D"), None)
    main = next((row for row in subsets if row["subset_id"] == "main_only"), None)
    all_four_avg_tokens = float(all_four["avg_tokens"]) if all_four else 0.0

    replay_rows = _load_replay(replay_path)
    factorial_rows: list[FactorialResult] = []
    if factorial_path.exists():
        factorial_rows = [
            FactorialResult.model_validate(row) for row in read_jsonl(factorial_path)
        ]

    # CMAB-budget-matched random pruning (per the brief's intent).
    cmab_random_baseline: dict | None = None
    cmab_size: float | None = None
    if replay_rows and factorial_rows:
        cmab_size = cmab_avg_subset_size(replay_rows)
        cmab_random_summary = baseline_summary(
            factorial_rows, random_target_size=cmab_size
        )
        cmab_random_baseline = cmab_random_summary["random_budget_matched"]

    policies = _replay_policy_summary(replay_rows)
    selected_subsets = {row["selected_subset_id"] for row in replay_rows}
    oracle = baselines.get("oracle_fixed_subset", {}) or {}
    static_bl = baselines.get("static_pruning", {}) or {}
    random_bl = baselines.get("random_budget_matched", {}) or {}

    # Self-consistency
    sc_rows = read_jsonl(sc_path) if sc_path.exists() else []
    sc_summary = self_consistency_summary(
        sc_rows, all_four_avg_tokens=all_four_avg_tokens
    )

    # Non-inferiority of CMAB vs all-four
    non_inf = (
        non_inferiority_report(factorial_rows, replay_rows)
        if factorial_rows and replay_rows
        else {"status": "insufficient_data"}
    )

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

    lines.extend(["", "## Baselines"])
    static_sid = (
        static_bl.get("subset_id", "n/a") if isinstance(static_bl, dict) else "n/a"
    )
    random_target = (
        random_bl.get("target_subset_id", "n/a")
        if isinstance(random_bl, dict)
        else "n/a"
    )
    oracle_sid = oracle.get("subset_id", "n/a") if isinstance(oracle, dict) else "n/a"
    lines.append(_format_baseline_line(f"Static pruning ({static_sid})", static_bl))
    random_size_label = (
        random_bl.get("target_size", "n/a") if isinstance(random_bl, dict) else "n/a"
    )
    lines.append(
        _format_baseline_line(
            f"Random budget-matched (target {random_target}, size={random_size_label})",
            random_bl,
        )
    )
    if cmab_random_baseline is not None:
        lines.append(
            _format_baseline_line(
                f"Random budget-matched to CMAB (size≈{cmab_size:.2f})",
                cmab_random_baseline,
            )
        )
    lines.append(
        f"- Oracle fixed-subset (MVP only): subset={oracle_sid}, "
        f"utility={_fmt(oracle, 'utility')}, "
        f"accuracy={_fmt(oracle, 'accuracy')}, "
        f"avg_tokens={_fmt(oracle, 'avg_tokens')}"
    )

    lines.extend(
        [
            "",
            "## CMAB Replay",
            "Bandit replay observes only the selected subset outcome at each step.",
        ]
    )
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
            f"- Subset space explored: {len(selected_subsets)}/16 "
            f"({len(selected_subsets) / 16:.2%})"
        )
        if replay_rows:
            lines.append(
                f"- CMAB avg subset size: {cmab_avg_subset_size(replay_rows):.2f}, "
                f"avg tokens/question: {cmab_avg_tokens(replay_rows):.1f}"
            )
    else:
        lines.append("- No bandit replay artifacts found.")

    lines.extend(["", "## CMAB vs All-Four: Non-Inferiority"])
    if non_inf.get("status") in {"insufficient_data", "no_overlap"}:
        lines.append(
            f"- Skipped: {non_inf['status']}. Run `replay-bandit` and "
            "`run-factorial` over a non-empty dataset to populate this section."
        )
    else:
        ci_low, ci_high = non_inf["accuracy_gap_ci"]
        lines.append(
            f"- N questions paired: {non_inf['n_questions']} "
            f"(bootstrap samples={non_inf['bootstrap_samples']})"
        )
        lines.append(
            f"- Accuracy all-four: {non_inf['accuracy_all_four']:.3f}, "
            f"CMAB ensemble: {non_inf['accuracy_cmab']:.3f}"
        )
        lines.append(
            f"- Accuracy gap (all_four − cmab): {non_inf['accuracy_gap']:+.3f} "
            f"95% CI [{ci_low:+.3f}, {ci_high:+.3f}], "
            f"ε={non_inf['epsilon']}, "
            f"non-inferior={non_inf['non_inferior']} "
            f"(CI-upper-bound: {non_inf['non_inferior_ci_upper']})"
        )
        mc = non_inf["mcnemar"]
        lines.append(
            f"- McNemar (threshold={non_inf['mcnemar_threshold']:.2f}): "
            f"all-four-correct/CMAB-wrong={mc['all_four_correct_cmab_wrong']}, "
            f"all-four-wrong/CMAB-correct={mc['all_four_wrong_cmab_correct']}"
        )

    lines.extend(["", "## Self-Consistency"])
    if sc_summary:
        for entry in sc_summary:
            lines.append(
                f"- {entry['label']}: n={entry['n']}, "
                f"accuracy={entry['accuracy']:.3f}, "
                f"avg_tokens={entry['avg_tokens']:.1f}, "
                f"utility={entry['utility']:.3f}"
            )
    else:
        lines.append(
            "- No self-consistency artifacts found. Run "
            "`gpqa-cmab run-self-consistency` to populate this section."
        )

    lines.extend(
        [
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
