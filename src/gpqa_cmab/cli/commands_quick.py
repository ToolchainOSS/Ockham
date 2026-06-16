"""The cheap end-to-end ``quick-check`` and ``smoke-test`` subcommands.

``quick-check`` runs either a single subset (2-5 calls) or the full 16-subset
factorial sweep on one sampled question; ``smoke-test`` exercises the whole
mock pipeline (factorial → replay → self-consistency → report) for free.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from gpqa_cmab.agents.main_integrator import run_main_integrator
from gpqa_cmab.agents.subagents import run_subagent
from gpqa_cmab.cli.support import (
    _apply_cli_overrides,
    _build_cost_guard,
    _cost_breakdown_for_rows,
    _file_logging,
    _manifest_argv,
    _pick_question,
    _preflight_real_llm,
    _progress,
    _resolve_provider,
    _settings_manifest,
    _setup_verbose_logging,
    _utc_now,
    make_client,
)
from gpqa_cmab.config import get_settings
from gpqa_cmab.dataset import load_questions
from gpqa_cmab.experiments.factorial import run_full_factorial
from gpqa_cmab.experiments.replay import replay_bandit
from gpqa_cmab.experiments.self_consistency import run_self_consistency_experiment
from gpqa_cmab.llm.mock import MockLLMClient
from gpqa_cmab.reporting import write_evaluation_outputs, write_report
from gpqa_cmab.telemetry import TelemetryLogger, write_jsonl, write_run_manifest

if TYPE_CHECKING:
    import argparse

    from gpqa_cmab.schemas import GPQAQuestion


def _quick_check_single_subset(
    args: argparse.Namespace,
    question: GPQAQuestion,
    provider: str,
    forced_mock: bool,
    verbose: int,
) -> None:
    """Cheap 2-5 call mode for narrow debugging of a single subset."""
    started_utc = _utc_now()
    settings = get_settings()
    client = make_client(provider)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "single_subset_trace.jsonl"
    log_path = output_dir / "single_subset.log"
    trace = TelemetryLogger(trace_path)
    subset = "".join(dict.fromkeys(ch.upper() for ch in args.subset if ch.strip()))
    if not subset or any(ch not in "ABCD" for ch in subset):
        raise SystemExit(
            f"--subset must be a non-empty string of letters from A,B,C,D "
            f"(got {args.subset!r})."
        )
    _preflight_real_llm(settings, planned_calls=len(subset) + 1)

    _progress(
        verbose,
        f"mode=single-subset subset={subset} provider={provider} "
        f"reasoning_effort={settings.reasoning_effort or 'off'}",
    )

    experiment = "quick-check"
    reports = {}
    for index, agent in enumerate(subset, start=1):
        _progress(
            verbose,
            f"step {index}/{len(subset) + 1}: calling subagent {agent} ...",
        )
        started = time.perf_counter()
        with _file_logging(log_path, settings.log_level):
            report, row = run_subagent(
                client,
                question,
                agent,
                experiment_id=experiment,
                model=settings.subagent_model,
                telemetry=trace,
            )
        reports[agent] = report
        elapsed = (time.perf_counter() - started) * 1000
        _progress(
            verbose,
            f"  subagent {agent} done in {elapsed:.0f}ms "
            f"tokens={row.usage.total_tokens}",
        )

    _progress(
        verbose,
        f"step {len(subset) + 1}/{len(subset) + 1}: calling main integrator ...",
    )
    started = time.perf_counter()
    with _file_logging(log_path, settings.log_level):
        main_output, main_row = run_main_integrator(
            client,
            question,
            {agent: reports[agent] for agent in subset},
            experiment_id=experiment,
            model=settings.main_model,
            telemetry=trace,
        )
    elapsed = (time.perf_counter() - started) * 1000
    _progress(
        verbose,
        f"  main integrator done in {elapsed:.0f}ms "
        f"tokens={main_row.usage.total_tokens} answer={main_output.final_answer}",
    )

    all_rows = trace.records
    total_tokens = sum(row.usage.total_tokens for row in all_rows)
    cost_breakdown = _cost_breakdown_for_rows(all_rows, settings)
    summary = {
        "ok": True,
        "mode": "single-subset",
        "provider": provider,
        "forced_mock": forced_mock,
        "reasoning_effort": settings.reasoning_effort,
        "question_id": question.question_id,
        "domain": question.domain,
        "subset": subset,
        "predicted_answer": main_output.final_answer,
        "correct_answer": question.correct_answer,
        "correct": main_output.final_answer == question.correct_answer,
        "confidence": main_output.confidence,
        "api_calls": len(all_rows),
        "tokens": {
            "prompt": sum(row.usage.prompt_tokens for row in all_rows),
            "completion": sum(row.usage.completion_tokens for row in all_rows),
            "total": total_tokens,
        },
        "estimated_cost_usd": cost_breakdown["estimated_cost_usd"],
        "cost_breakdown": cost_breakdown,
        "latency_ms_total": sum(row.latency_ms for row in all_rows),
    }
    summary_path = output_dir / "single_subset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path = output_dir / "single_subset_manifest.json"
    write_run_manifest(
        manifest_path,
        command="quick-check --subset",
        argv=_manifest_argv(args),
        started_utc=started_utc,
        status="completed",
        inputs=[args.input],
        artifacts=[summary_path, log_path],
        traces=[trace_path],
        settings=_settings_manifest(settings),
        extra={
            "question_id": question.question_id,
            "subset": subset,
            "provider": provider,
            "forced_mock": forced_mock,
        },
    )
    summary["output_dir"] = str(output_dir)
    summary["trace"] = str(trace_path)
    summary["log"] = str(log_path)
    summary["manifest"] = str(manifest_path)
    print(json.dumps(summary, indent=2))


def _quick_check_factorial(
    args: argparse.Namespace,
    question: GPQAQuestion,
    provider: str,
    forced_mock: bool,
    verbose: int,
) -> None:
    """Default mode: run the full 16-subset factorial on the single question.

    This treats the sampled physics question as if it were the entire
    experiment. Useful for shaking out the whole pipeline (subagents + every
    main-integrator subset) in one cheap call.
    """
    settings = get_settings()
    client = make_client(provider)
    output_dir: Path = args.output_dir
    started_utc = _utc_now()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "full_factorial_trace.jsonl"
    log_path = output_dir / "quick_check.log"
    trace = TelemetryLogger(trace_path)

    _progress(
        verbose,
        f"mode=factorial provider={provider} "
        f"subagent_model={settings.subagent_model} "
        f"main_model={settings.main_model} "
        f"reasoning_effort={settings.reasoning_effort or 'off'}",
    )
    _progress(
        verbose,
        "planned LLM calls: 4 subagents + 16 main-integrator subsets = 20",
    )

    started = time.perf_counter()
    _preflight_real_llm(settings, planned_calls=20)
    guard = _build_cost_guard(args, settings)
    with _file_logging(log_path, settings.log_level):
        results = run_full_factorial(
            [question],
            client,
            main_model=settings.main_model,
            subagent_model=settings.subagent_model,
            experiment_id="quick-check",
            cost_guard=guard,
            telemetry=trace,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000

    if len(results) != 16:
        # Should not happen for a single complete question, but be defensive.
        _progress(
            verbose,
            f"WARNING: expected 16 factorial rows, got {len(results)}.",
        )

    # Persist artifacts so users can inspect / re-run downstream commands.
    factorial_path = output_dir / "full_factorial_results.jsonl"
    write_jsonl(factorial_path, results)
    write_evaluation_outputs(results, output_dir)

    # Per-subset table.
    per_subset = [
        {
            "subset_id": row.subset_id,
            "selected": row.selected_subagents,
            "predicted": row.final_answer,
            "correct": row.correct,
            "tokens": row.usage.total_tokens,
            "confidence": row.confidence,
        }
        for row in results
    ]
    correct_subsets = [row.subset_id for row in results if row.correct]
    total_prompt_tokens = sum(row.usage.prompt_tokens for row in trace.records)
    total_completion_tokens = sum(row.usage.completion_tokens for row in trace.records)
    total_tokens = sum(row.usage.total_tokens for row in trace.records)

    full_row = next((row for row in results if row.subset_id == "A,B,C,D"), None)
    full_predicted = full_row.final_answer if full_row else None
    full_correct = bool(full_row and full_row.correct)
    num_correct = sum(1 for row in results if row.correct)

    cost_breakdown = _cost_breakdown_for_rows(trace.records, settings)
    estimated_cost = cost_breakdown["estimated_cost_usd"]
    summary = {
        "ok": True,
        "mode": "factorial",
        "provider": provider,
        "forced_mock": forced_mock,
        "reasoning_effort": settings.reasoning_effort,
        "question_id": question.question_id,
        "domain": question.domain,
        "correct_answer": question.correct_answer,
        "subsets_evaluated": len(results),
        "subsets_correct": num_correct,
        "subset_accuracy": num_correct / max(len(results), 1),
        "full_subset_predicted": full_predicted,
        "full_subset_correct": full_correct,
        "correct_subset_ids": correct_subsets,
        "api_calls": len(trace.records),
        "tokens": {
            "prompt": total_prompt_tokens,
            "completion": total_completion_tokens,
            "total": total_tokens,
        },
        "estimated_cost_usd": estimated_cost,
        "cost_breakdown": cost_breakdown,
        "wall_time_ms": int(elapsed_ms),
        "output_dir": str(output_dir),
        "trace": str(trace_path),
        "log": str(log_path),
        "per_subset": per_subset,
    }
    summary_path = output_dir / "quick_check_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path = output_dir / "quick_check_manifest.json"
    write_run_manifest(
        manifest_path,
        command="quick-check",
        argv=_manifest_argv(args),
        started_utc=started_utc,
        status="completed",
        inputs=[args.input],
        artifacts=[
            factorial_path,
            output_dir / "subset_accuracy_table.csv",
            output_dir / "metrics_summary.json",
            summary_path,
            log_path,
        ],
        traces=[trace_path],
        settings=_settings_manifest(settings),
        budget=guard.snapshot(),
        extra={
            "question_id": question.question_id,
            "provider": provider,
            "forced_mock": forced_mock,
        },
    )
    summary["manifest"] = str(manifest_path)
    print(json.dumps(summary, indent=2))


def cmd_quick_check(args: argparse.Namespace) -> None:
    """Cheap end-to-end pipeline sanity check on one random question.

    Default mode runs the full 16-subset factorial sweep on a single sampled
    physics question (4 subagent + 16 main-integrator = 20 LLM calls). Pass
    `--subset X` to instead run a single subset for the cheapest possible
    debug path (2 calls when X='A').

    Defaults to the mock provider so it costs nothing. To exercise a real
    provider, pass `--allow-real-llm` AND set `LLM_PROVIDER` to a non-mock
    value; otherwise the command forces mock mode.
    """
    _apply_cli_overrides(args)
    verbose = int(getattr(args, "verbose", 0) or 0)
    _setup_verbose_logging(verbose)

    question = _pick_question(args)
    provider, forced_mock = _resolve_provider(args.allow_real_llm)
    _progress(
        verbose,
        f"picked question_id={question.question_id} domain={question.domain} "
        f"correct={question.correct_answer}",
    )
    if forced_mock:
        _progress(
            verbose,
            "WARNING: --allow-real-llm not set; forcing mock provider.",
        )

    if args.subset is not None:
        _quick_check_single_subset(args, question, provider, forced_mock, verbose)
    else:
        _quick_check_factorial(args, question, provider, forced_mock, verbose)


def cmd_smoke_test(args: argparse.Namespace) -> None:
    started_utc = _utc_now()
    mock_flag = bool(getattr(args, "mock", False))
    sample = Path("artifacts/cache/mock_smoke.jsonl")
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text(
        json.dumps(
            {
                "question_id": "mock-1",
                "domain": "physics",
                "question": "Which option is correct in this mock physics question?",
                "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
                "correct_answer": "A",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    results_path = Path("artifacts/results/full_factorial_results.jsonl")
    trace_path = Path("artifacts/results/smoke_trace.jsonl")
    log_path = Path("artifacts/results/smoke.log")
    trace = TelemetryLogger(trace_path)
    client = MockLLMClient()
    questions = load_questions(sample, "physics")
    with _file_logging(log_path, get_settings().log_level):
        rows = run_full_factorial(
            questions,
            client,
            main_model="mock-main",
            subagent_model="mock-subagent",
            experiment_id="mock-smoke",
            telemetry=trace,
        )
    write_jsonl(results_path, rows)
    write_evaluation_outputs(rows, Path("artifacts/results"))
    steps = replay_bandit(rows, policy="superarm-ts", seeds=2)
    write_jsonl(Path("artifacts/results/bandit_replay_results.jsonl"), steps)
    bootstrap_path = Path("artifacts/results/bootstrap_results.json")
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_path.write_text(
        json.dumps({"status": "available"}, indent=2), encoding="utf-8"
    )
    summary_path = Path("artifacts/results/bandit_summary.json")
    summary_path.write_text(
        json.dumps(
            {"policy": "superarm-ts", "seeds": 2, "partial_information": True},
            indent=2,
        ),
        encoding="utf-8",
    )
    with _file_logging(log_path, get_settings().log_level):
        sc_rows = run_self_consistency_experiment(
            questions,
            client,
            model="mock-self-consistency",
            k_values=[1, 4],
            seed=0,
            telemetry=trace,
        )
    write_jsonl(Path("artifacts/results/self_consistency_results.jsonl"), sc_rows)
    write_report(Path("artifacts/results"), Path("artifacts/reports/mvp_report.md"))
    manifest_path = Path("artifacts/results/smoke_manifest.json")
    write_run_manifest(
        manifest_path,
        command="smoke-test",
        argv=_manifest_argv(args),
        started_utc=started_utc,
        status="completed",
        inputs=[sample],
        artifacts=[
            results_path,
            Path("artifacts/results/subset_accuracy_table.csv"),
            Path("artifacts/results/metrics_summary.json"),
            Path("artifacts/results/bandit_replay_results.jsonl"),
            Path("artifacts/results/self_consistency_results.jsonl"),
            Path("artifacts/reports/mvp_report.md"),
            log_path,
        ],
        traces=[trace_path],
        settings=_settings_manifest(get_settings()),
        extra={
            "provider": "mock",
            "mock_flag": mock_flag,
            "factorial_rows": len(rows),
            "bandit_steps": len(steps),
        },
    )
    print(
        json.dumps(
            {
                "ok": True,
                "provider": "mock",
                "mock_flag": mock_flag,
                "factorial_rows": len(rows),
                "bandit_steps": len(steps),
                "trace": str(trace_path),
                "manifest": str(manifest_path),
            }
        )
    )
