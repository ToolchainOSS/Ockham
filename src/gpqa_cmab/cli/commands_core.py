"""Core experiment subcommands: data validation, subagent cache building,
the full factorial sweep, evaluation, bandit replay, the offline CMAB
benchmark, the markdown report, self-consistency, and baselines.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from gpqa_cmab.agents.subagents import run_all_subagents
from gpqa_cmab.cli.support import (
    _apply_cli_overrides,
    _build_cost_guard,
    _file_logging,
    _log_path,
    _manifest_argv,
    _manifest_path,
    _preflight_real_llm,
    _require_questions,
    _settings_manifest,
    _trace_path,
    _utc_now,
    make_client,
)
from gpqa_cmab.config import get_settings
from gpqa_cmab.cost_guard import BudgetExceeded
from gpqa_cmab.dataset import load_questions
from gpqa_cmab.experiments.cmab_benchmark import report_to_jsonable, run_benchmark
from gpqa_cmab.experiments.factorial import load_subagent_cache, run_full_factorial
from gpqa_cmab.experiments.replay import replay_bandit
from gpqa_cmab.experiments.self_consistency import run_self_consistency_experiment
from gpqa_cmab.metrics import baseline_summary
from gpqa_cmab.reporting import write_evaluation_outputs, write_report
from gpqa_cmab.schemas import AgentId, FactorialResult
from gpqa_cmab.telemetry import (
    TelemetryLogger,
    read_jsonl,
    write_jsonl,
    write_run_manifest,
)

if TYPE_CHECKING:
    import argparse


def cmd_validate_data(args: argparse.Namespace) -> None:
    questions = load_questions(args.input, args.domain, args.max_questions)
    print(json.dumps({"domain": args.domain, "questions": len(questions)}))


def cmd_run_subagents(args: argparse.Namespace) -> None:
    started_utc = _utc_now()
    _apply_cli_overrides(args)
    questions = load_questions(args.input, args.domain, args.max_questions)
    _require_questions(questions, input_path=args.input, domain=args.domain)
    settings = get_settings()
    _preflight_real_llm(settings, planned_calls=len(questions) * 4)
    guard = _build_cost_guard(args, settings)
    client = make_client(settings.llm_provider)
    experiment_id = f"subagent-cache-{uuid.uuid4()}"
    trace_path = _trace_path(args.output)
    log_path = _log_path(args.output)
    trace = TelemetryLogger(trace_path)
    rows = []
    with _file_logging(log_path, settings.log_level):
        for question in questions:
            if guard.would_exceed_calls(4) or guard.exhausted():
                break
            try:
                start_index = len(trace.records)
                reports, telemetry_rows = run_all_subagents(
                    client,
                    question,
                    experiment_id=experiment_id,
                    model=settings.subagent_model,
                    telemetry=trace,
                )
            except BudgetExceeded:
                break
            for telem_row in trace.records_since(start_index):
                guard.add_call_usage(telem_row.usage)
            telemetry_by_agent = {
                AgentId(row.agent_type): row for row in telemetry_rows
            }
            for agent, report in reports.items():
                rows.append(
                    {
                        "question_id": question.question_id,
                        "agent": agent,
                        "report": report.model_dump(mode="json"),
                        "telemetry": telemetry_by_agent[agent].model_dump(mode="json"),
                    }
                )
    write_jsonl(args.output, rows)
    manifest_path = _manifest_path(args.output)
    write_run_manifest(
        manifest_path,
        command="run-subagents",
        argv=_manifest_argv(args),
        started_utc=started_utc,
        status="completed",
        inputs=[args.input],
        artifacts=[args.output, log_path],
        traces=[trace_path],
        settings=_settings_manifest(settings),
        budget=guard.snapshot(),
        extra={"experiment_id": experiment_id, "rows": len(rows)},
    )
    print(
        json.dumps(
            {
                "cached_reports": len(rows),
                "output": str(args.output),
                "trace": str(trace_path),
                "log": str(log_path),
                "manifest": str(manifest_path),
                "budget": guard.snapshot(),
            }
        )
    )


def cmd_run_factorial(args: argparse.Namespace) -> None:
    started_utc = _utc_now()
    _apply_cli_overrides(args)
    questions = load_questions(args.input, args.domain, args.max_questions)
    _require_questions(questions, input_path=args.input, domain=args.domain)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "questions": len(questions),
                    "estimated_main_calls": len(questions) * 16,
                }
            )
        )
        return
    settings = get_settings()
    subagent_cache = None
    if args.subagent_cache is not None and args.subagent_cache.exists():
        subagent_cache = load_subagent_cache(read_jsonl(args.subagent_cache))
    planned = len(questions) * (16 if subagent_cache is not None else 20)
    _preflight_real_llm(settings, planned_calls=planned)
    guard = _build_cost_guard(args, settings)
    trace_path = _trace_path(args.output)
    log_path = _log_path(args.output)
    trace = TelemetryLogger(trace_path)
    with _file_logging(log_path, settings.log_level):
        results = run_full_factorial(
            questions,
            make_client(settings.llm_provider),
            main_model=settings.main_model,
            subagent_model=settings.subagent_model,
            subagent_cache=subagent_cache,
            cost_guard=guard,
            telemetry=trace,
        )
    write_jsonl(args.output, results)
    manifest_path = _manifest_path(args.output)
    manifest_inputs = [args.input]
    if args.subagent_cache is not None:
        manifest_inputs.append(args.subagent_cache)
    write_run_manifest(
        manifest_path,
        command="run-factorial",
        argv=_manifest_argv(args),
        started_utc=started_utc,
        status="completed",
        inputs=manifest_inputs,
        artifacts=[args.output, log_path],
        traces=[trace_path],
        settings=_settings_manifest(settings),
        budget=guard.snapshot(),
        extra={
            "rows": len(results),
            "used_subagent_cache": subagent_cache is not None,
            "domain": args.domain,
            "max_questions": args.max_questions,
        },
    )
    print(
        json.dumps(
            {
                "rows": len(results),
                "output": str(args.output),
                "trace": str(trace_path),
                "log": str(log_path),
                "manifest": str(manifest_path),
                "used_subagent_cache": subagent_cache is not None,
                "budget": guard.snapshot(),
            }
        )
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    started_utc = _utc_now()
    _apply_cli_overrides(args)
    settings = get_settings()
    rows = [FactorialResult.model_validate(row) for row in read_jsonl(args.results)]
    write_evaluation_outputs(
        rows,
        args.output_dir,
        lambda_token=settings.lambda_token,
        lambda_call=settings.lambda_call,
    )
    artifacts = [
        args.output_dir / "subset_accuracy_table.csv",
        args.output_dir / "metrics_summary.json",
    ]
    manifest_path = args.output_dir / "evaluate_manifest.json"
    write_run_manifest(
        manifest_path,
        command="evaluate",
        argv=_manifest_argv(args),
        started_utc=started_utc,
        status="completed",
        inputs=[args.results],
        artifacts=artifacts,
        settings=_settings_manifest(settings),
        extra={"rows": len(rows)},
    )
    print(
        json.dumps({"output_dir": str(args.output_dir), "manifest": str(manifest_path)})
    )


def cmd_replay_bandit(args: argparse.Namespace) -> None:
    started_utc = _utc_now()
    _apply_cli_overrides(args)
    settings = get_settings()
    rows = [FactorialResult.model_validate(row) for row in read_jsonl(args.results)]
    steps = replay_bandit(
        rows,
        policy=args.policy,
        seeds=args.seeds,
        lambda_token=settings.lambda_token,
        lambda_call=settings.lambda_call,
    )
    write_jsonl(args.output, steps)
    manifest_path = _manifest_path(args.output)
    write_run_manifest(
        manifest_path,
        command="replay-bandit",
        argv=_manifest_argv(args),
        started_utc=started_utc,
        status="completed",
        inputs=[args.results],
        artifacts=[args.output],
        settings=_settings_manifest(settings),
        extra={"steps": len(steps), "policy": args.policy, "seeds": args.seeds},
    )
    print(
        json.dumps(
            {
                "steps": len(steps),
                "output": str(args.output),
                "manifest": str(manifest_path),
            }
        )
    )


def cmd_benchmark_cmab(args: argparse.Namespace) -> None:
    started_utc = _utc_now()
    settings = get_settings()
    report = run_benchmark(
        args.metrics_summary,
        n_seeds=args.seeds,
        n_steps=args.steps,
        lambda_token=settings.lambda_token,
        lambda_call=settings.lambda_call,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report_to_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path = _manifest_path(args.output)
    write_run_manifest(
        manifest_path,
        command="benchmark-cmab",
        argv=_manifest_argv(args),
        started_utc=started_utc,
        status="completed",
        inputs=[args.metrics_summary] if args.metrics_summary else [],
        artifacts=[args.output],
        settings=_settings_manifest(settings),
        extra={
            "seeds": args.seeds,
            "steps": args.steps,
            "source": (
                str(args.metrics_summary)
                if args.metrics_summary
                else "mvp_aggregates (canonical 86Q baseline)"
            ),
        },
    )
    # Print a compact comparison table for the terminal.
    rows = [
        {
            "policy": p.name,
            "accuracy": round(p.accuracy_mean, 4),
            "tokens": round(p.tokens_mean, 1),
            "utility": round(p.utility_mean, 4),
            "unique_subsets": round(p.unique_subsets_mean, 2),
        }
        for p in report.policies
    ]
    for name, s in report.static_baselines.items():
        rows.append(
            {
                "policy": f"static[{name}]",
                "accuracy": round(s["accuracy_mean"], 4),
                "tokens": round(s["avg_tokens"], 1),
                "utility": round(s["utility_mean"], 4),
                "unique_subsets": 1,
            }
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest": str(manifest_path),
                "seeds": args.seeds,
                "steps": args.steps,
                "comparison": rows,
            },
            indent=2,
        )
    )


def cmd_report(args: argparse.Namespace) -> None:
    started_utc = _utc_now()
    write_report(args.results_dir, args.output)
    manifest_path = _manifest_path(args.output)
    write_run_manifest(
        manifest_path,
        command="report",
        argv=_manifest_argv(args),
        started_utc=started_utc,
        status="completed",
        inputs=[
            args.results_dir / "metrics_summary.json",
            args.results_dir / "bandit_replay_results.jsonl",
            args.results_dir / "self_consistency_results.jsonl",
        ],
        artifacts=[args.output],
        settings=_settings_manifest(get_settings()),
    )
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path)}))


def cmd_run_self_consistency(args: argparse.Namespace) -> None:
    started_utc = _utc_now()
    _apply_cli_overrides(args)
    questions = load_questions(args.input, args.domain, args.max_questions)
    _require_questions(questions, input_path=args.input, domain=args.domain)
    settings = get_settings()
    k_values = [int(value) for value in args.k_values.split(",") if value.strip()]
    planned = len(questions) * sum(k_values)
    _preflight_real_llm(settings, planned_calls=planned)
    guard = _build_cost_guard(args, settings)
    trace_path = _trace_path(args.output)
    log_path = _log_path(args.output)
    trace = TelemetryLogger(trace_path)
    with _file_logging(log_path, settings.log_level):
        rows = run_self_consistency_experiment(
            questions,
            make_client(settings.llm_provider),
            model=settings.self_consistency_model,
            k_values=k_values,
            seed=args.seed,
            temperature=args.temperature,
            cost_guard=guard,
            telemetry=trace,
        )
    write_jsonl(args.output, rows)
    manifest_path = _manifest_path(args.output)
    write_run_manifest(
        manifest_path,
        command="run-self-consistency",
        argv=_manifest_argv(args),
        started_utc=started_utc,
        status="completed",
        inputs=[args.input],
        artifacts=[args.output, log_path],
        traces=[trace_path],
        settings=_settings_manifest(settings),
        budget=guard.snapshot(),
        extra={"rows": len(rows), "k_values": k_values, "seed": args.seed},
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "output": str(args.output),
                "trace": str(trace_path),
                "log": str(log_path),
                "manifest": str(manifest_path),
                "budget": guard.snapshot(),
            }
        )
    )


def cmd_baselines(args: argparse.Namespace) -> None:
    started_utc = _utc_now()
    _apply_cli_overrides(args)
    settings = get_settings()
    rows = [FactorialResult.model_validate(row) for row in read_jsonl(args.results)]
    summary = baseline_summary(
        rows,
        static_subset_id=args.static_subset,
        random_target_subset_id=args.target_subset,
        random_target_size=args.target_size,
        seeds=args.seeds,
        lambda_token=settings.lambda_token,
        lambda_call=settings.lambda_call,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path = _manifest_path(args.output)
    write_run_manifest(
        manifest_path,
        command="baselines",
        argv=_manifest_argv(args),
        started_utc=started_utc,
        status="completed",
        inputs=[args.results],
        artifacts=[args.output],
        settings=_settings_manifest(settings),
        extra={"static_subset": args.static_subset, "seeds": args.seeds},
    )
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path)}))
