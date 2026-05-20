from __future__ import annotations

import csv
import json
from pathlib import Path

from gpqa_cmab.metrics import subset_table
from gpqa_cmab.schemas import FactorialResult


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
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def write_report(results_dir: Path, output: Path) -> None:
    summary_path = results_dir / "metrics_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    subsets = summary.get("subsets", [])
    all_four = next((row for row in subsets if row["subset_id"] == "A,B,C,D"), None)
    main = next((row for row in subsets if row["subset_id"] == "main_only"), None)
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
    lines.extend(
        [
            "",
            "## CMAB And Baselines",
            "Bandit replay outputs are written separately and obey partial "
            "information: each step observes only the selected subset row.",
            "Static pruning, random pruning, oracle fixed-subset reference, "
            "and self-consistency are supported by the metrics/replay artifacts "
            "or CLI extension points.",
            "",
            "## Limitations",
            "Mock mode is deterministic and validates orchestration only. "
            "Real claims require live model runs, repeated seeds, and bootstrap "
            "intervals over enough questions.",
            "",
            "## Next Steps",
            "Run live provider experiments with cost caps, add richer uncertainty "
            "reporting, and scale beyond four optional subagents.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(row: dict | None, key: str) -> str:
    if not row:
        return "n/a"
    value = row[key]
    return f"{value:.3f}" if isinstance(value, float) else str(value)
