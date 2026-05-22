from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from gpqa_cmab.agents.main_integrator import run_main_integrator
from gpqa_cmab.agents.subagents import run_all_subagents, run_subagent
from gpqa_cmab.config import Settings, clear_settings_cache, get_settings, load_dotenv
from gpqa_cmab.cost_guard import (
    BudgetExceeded,
    CostGuard,
    CostRates,
    usage_cost_breakdown,
)
from gpqa_cmab.dataset import load_questions
from gpqa_cmab.experiments.factorial import load_subagent_cache, run_full_factorial
from gpqa_cmab.experiments.replay import replay_bandit
from gpqa_cmab.experiments.self_consistency import run_self_consistency_experiment
from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.llm.mock import MockLLMClient
from gpqa_cmab.llm.openai_compatible import (
    AzureOpenAIClient,
    OpenAICompatibleClient,
)
from gpqa_cmab.metrics import baseline_summary
from gpqa_cmab.reporting import write_evaluation_outputs, write_report
from gpqa_cmab.schemas import FactorialResult, GPQAQuestion
from gpqa_cmab.telemetry import (
    TelemetryLogger,
    read_jsonl,
    write_jsonl,
    write_run_manifest,
)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    args._gpqa_argv = [parser.prog, *raw_argv]
    if getattr(args, "env_file", None) is not None:
        loaded = load_dotenv(args.env_file, override=True)
        if loaded is None:
            raise SystemExit(f"--env-file not found: {args.env_file}")
        clear_settings_cache()
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpqa-cmab")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            "Path to a .env file to load before running the command. By "
            "default the nearest .env walking up from the current directory "
            "is used automatically; this flag overrides that and takes "
            "precedence over already-set environment variables."
        ),
    )
    sub = parser.add_subparsers(required=True)

    def _add_llm_overrides(p: argparse.ArgumentParser, *, models: tuple[str, ...]):
        """Attach the env-mirrored LLM/runtime override flags shared by every
        command that issues LLM calls. ``models`` controls which model-name
        overrides are exposed (main / subagent / self_consistency)."""
        if "main" in models:
            p.add_argument(
                "--main-model",
                default=None,
                help="Override MAIN_MODEL for this run.",
            )
        if "subagent" in models:
            p.add_argument(
                "--subagent-model",
                default=None,
                help="Override SUBAGENT_MODEL for this run.",
            )
        if "self_consistency" in models:
            p.add_argument(
                "--self-consistency-model",
                default=None,
                help="Override SELF_CONSISTENCY_MODEL for this run.",
            )
        p.add_argument(
            "--reasoning-effort",
            default=None,
            help=("Override REASONING_EFFORT (none|minimal|low|medium|high|xhigh)."),
        )
        p.add_argument(
            "--max-output-tokens",
            type=int,
            default=None,
            help=(
                "Override MAX_OUTPUT_TOKENS. CRITICAL when running reasoning "
                "models — without it, completions can spend tens of thousands "
                "of billed reasoning tokens per call."
            ),
        )
        p.add_argument(
            "--json-max-retries",
            type=int,
            default=None,
            help=(
                "Override LLM_JSON_MAX_RETRIES. Each retry is a billed LLM "
                "call; keep small (default 2)."
            ),
        )

    def _add_cost_caps(p: argparse.ArgumentParser):
        """Attach the run-wide cost / call cap flags."""
        p.add_argument(
            "--max-api-calls",
            type=int,
            default=None,
            help="Cap on billed LLM calls for this run.",
        )
        p.add_argument(
            "--max-estimated-cost-usd",
            type=float,
            default=None,
            help=(
                "Cap on cumulative estimated USD (requires at least one "
                "tiered pricing rate)."
            ),
        )
        p.add_argument(
            "--cost-input-usd-per-1m-tokens",
            type=float,
            default=None,
            help="Override COST_INPUT_USD_PER_1M_TOKENS for uncached input.",
        )
        p.add_argument(
            "--cost-cached-input-usd-per-1m-tokens",
            type=float,
            default=None,
            help="Override COST_CACHED_INPUT_USD_PER_1M_TOKENS.",
        )
        p.add_argument(
            "--cost-output-usd-per-1m-tokens",
            type=float,
            default=None,
            help="Override COST_OUTPUT_USD_PER_1M_TOKENS.",
        )

    def _add_lambdas(p: argparse.ArgumentParser):
        p.add_argument(
            "--lambda-token",
            type=float,
            default=None,
            help="Override LAMBDA_TOKEN (utility token weight).",
        )
        p.add_argument(
            "--lambda-call",
            type=float,
            default=None,
            help="Override LAMBDA_CALL (utility per-subagent-call weight).",
        )

    validate = sub.add_parser("validate-data")
    validate.add_argument("--input", required=True, type=Path)
    validate.add_argument("--domain", default="physics")
    validate.add_argument("--max-questions", type=int)
    validate.set_defaults(func=cmd_validate_data)

    run_subagents = sub.add_parser("run-subagents")
    run_subagents.add_argument("--input", required=True, type=Path)
    run_subagents.add_argument("--domain", default="physics")
    run_subagents.add_argument("--output", required=True, type=Path)
    run_subagents.add_argument("--max-questions", type=int)
    _add_cost_caps(run_subagents)
    _add_llm_overrides(run_subagents, models=("subagent",))
    run_subagents.set_defaults(func=cmd_run_subagents)

    factorial = sub.add_parser("run-factorial")
    factorial.add_argument("--input", required=True, type=Path)
    factorial.add_argument("--domain", default="physics")
    factorial.add_argument("--subagent-cache", type=Path)
    factorial.add_argument("--output", required=True, type=Path)
    factorial.add_argument("--max-questions", type=int)
    _add_cost_caps(factorial)
    _add_llm_overrides(factorial, models=("main", "subagent"))
    factorial.add_argument("--dry-run", action="store_true")
    factorial.set_defaults(func=cmd_run_factorial)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--results", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    _add_lambdas(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    replay = sub.add_parser("replay-bandit")
    replay.add_argument("--results", required=True, type=Path)
    replay.add_argument(
        "--policy", choices=["superarm-ts", "structured-cmab"], default="superarm-ts"
    )
    replay.add_argument("--seeds", type=int, default=10)
    replay.add_argument("--output", required=True, type=Path)
    _add_lambdas(replay)
    replay.set_defaults(func=cmd_replay_bandit)

    report = sub.add_parser("report")
    report.add_argument("--results-dir", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    report.set_defaults(func=cmd_report)

    sc = sub.add_parser("run-self-consistency")
    sc.add_argument("--input", required=True, type=Path)
    sc.add_argument("--domain", default="physics")
    sc.add_argument("--output", required=True, type=Path)
    sc.add_argument("--k-values", default="1,4,8,16")
    sc.add_argument("--max-questions", type=int)
    sc.add_argument("--seed", type=int, default=0)
    sc.add_argument("--temperature", type=float, default=0.7)
    _add_cost_caps(sc)
    _add_llm_overrides(sc, models=("self_consistency",))
    sc.set_defaults(func=cmd_run_self_consistency)

    baselines = sub.add_parser("baselines")
    baselines.add_argument("--results", required=True, type=Path)
    baselines.add_argument("--output", required=True, type=Path)
    baselines.add_argument("--static-subset", default="A")
    baselines.add_argument("--seeds", type=int, default=100)
    baselines.add_argument(
        "--target-subset",
        default="A,B,C,D",
        help="Subset whose average size random pruning will match.",
    )
    baselines.add_argument(
        "--target-size",
        type=float,
        default=None,
        help=(
            "Override the random-pruning target size (rounded to nearest int "
            "in [0,4]). Useful for budget-matching against CMAB."
        ),
    )
    _add_lambdas(baselines)
    baselines.set_defaults(func=cmd_baselines)

    smoke = sub.add_parser("smoke-test")
    smoke.add_argument(
        "--mock",
        action="store_true",
        help="Explicitly acknowledge mock mode; smoke-test is always local and free.",
    )
    smoke.set_defaults(func=cmd_smoke_test)

    quick = sub.add_parser(
        "quick-check",
        help=(
            "Run a single random physics question through the subagent + main "
            "integrator pipeline. Defaults to the mock provider so it is free."
        ),
    )
    quick.add_argument(
        "--input",
        type=Path,
        default=Path("data/gpqa_diamond.csv"),
        help="Dataset path (default: data/gpqa_diamond.csv).",
    )
    quick.add_argument("--domain", default="physics")
    quick.add_argument(
        "--subset",
        default=None,
        help=(
            "Optional: run ONLY the named subset (string of letters from "
            "{A,B,C,D}) for the ultra-cheap debug mode (1 subagent + 1 "
            "integrator call when subset='A'). If omitted, the command runs "
            "the full 16-subset factorial sweep on the sampled question "
            "(4 subagent + 16 integrator = 20 LLM calls)."
        ),
    )
    quick.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for picking the question. Omit for a fresh random pick.",
    )
    quick.add_argument(
        "--question-id",
        default=None,
        help="Pick a specific question by id instead of sampling randomly.",
    )
    quick.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/quick_check"),
        help=(
            "Where to write factorial JSONL + evaluation outputs in "
            "factorial mode. Ignored when --subset is provided."
        ),
    )
    quick.add_argument(
        "--allow-real-llm",
        action="store_true",
        help=(
            "Required to use a non-mock LLM_PROVIDER. Without this flag the "
            "command forces mock mode so it never burns billable tokens."
        ),
    )
    quick.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Stream progress to stderr while the pipeline runs so real-LLM "
            "calls don't look 'frozen'. Pass -v for per-step info, -vv for "
            "debug logging (full prompts and responses)."
        ),
    )
    _add_cost_caps(quick)
    _add_llm_overrides(quick, models=("main", "subagent"))
    quick.set_defaults(func=cmd_quick_check)
    return parser


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
            telemetry_by_agent = {row.agent_type: row for row in telemetry_rows}
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


def _setup_verbose_logging(verbose: int) -> None:
    if verbose >= 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s | %(message)s",
            stream=sys.stderr,
            force=True,
        )
    elif verbose == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s | %(message)s",
            stream=sys.stderr,
            force=True,
        )


def _pick_question(args: argparse.Namespace):
    questions = load_questions(args.input, args.domain)
    if not questions:
        raise SystemExit(f"No {args.domain!r} questions found in {args.input}.")
    if args.question_id is not None:
        matches = [q for q in questions if q.question_id == args.question_id]
        if not matches:
            raise SystemExit(f"Question id {args.question_id!r} not found.")
        return matches[0]
    rng = random.Random(args.seed)
    return rng.choice(questions)


def _resolve_provider(allow_real_llm: bool) -> tuple[str, bool]:
    settings = get_settings()
    if settings.llm_provider != "mock" and not allow_real_llm:
        return "mock", True
    return settings.llm_provider, False


def _quick_check_single_subset(args, question, provider, forced_mock, verbose):
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


def _quick_check_factorial(args, question, provider, forced_mock, verbose):
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


def _progress(verbose: int, message: str) -> None:
    """Stream a progress line to stderr so users see activity in real time."""
    if verbose <= 0:
        return
    print(f"[quick-check] {message}", file=sys.stderr, flush=True)


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


_PREFLIGHT_WARNED = False


def _is_real_provider(settings: Settings) -> bool:
    return settings.llm_provider != "mock"


def _preflight_real_llm(settings: Settings, *, planned_calls: int) -> None:
    """Emit loud safety warnings to stderr before a real-LLM run.

    Catches the three highest-cost foot-guns:
      1. ``MAX_OUTPUT_TOKENS`` unset (reasoning models can stream tens of
         thousands of billed reasoning tokens per call).
        2. No tiered pricing rate configured, which silently disables every USD
            cost cap downstream.
      3. No global ``MAX_TOTAL_COST_USD`` / ``MAX_TOTAL_API_CALLS`` ceiling
         configured for a sweep with a large planned-call budget.
    """
    global _PREFLIGHT_WARNED
    if not _is_real_provider(settings) or _PREFLIGHT_WARNED:
        return
    _PREFLIGHT_WARNED = True
    warnings: list[str] = []
    if settings.max_output_tokens is None:
        warnings.append(
            "MAX_OUTPUT_TOKENS is UNSET. Reasoning models can stream tens of "
            "thousands of billed tokens per call. Set MAX_OUTPUT_TOKENS to "
            "cap each completion."
        )
    if not _cost_rates_from_settings(settings).enabled:
        warnings.append(
            "No USD pricing is configured; every USD cost cap is INACTIVE. "
            "Set COST_INPUT_USD_PER_1M_TOKENS, "
            "COST_CACHED_INPUT_USD_PER_1M_TOKENS, and "
            "COST_OUTPUT_USD_PER_1M_TOKENS. If only one or two are set, "
            "missing rates are filled with the maximum configured rate."
        )
    if (
        settings.max_total_cost_usd is None
        and settings.max_total_api_calls is None
        and planned_calls > 100
    ):
        warnings.append(
            f"Run plans {planned_calls} LLM calls with no MAX_TOTAL_COST_USD "
            "or MAX_TOTAL_API_CALLS ceiling. Set one to bound the worst-case "
            "bill."
        )
    if warnings:
        print(
            f"[gpqa-cmab] COST SAFETY: provider={settings.llm_provider} "
            f"planned_calls~{planned_calls}",
            file=sys.stderr,
            flush=True,
        )
        for line in warnings:
            print(f"  ! {line}", file=sys.stderr, flush=True)


def _cost_rates_from_settings(settings: Settings) -> CostRates:
    return CostRates(
        input_usd_per_1m_tokens=settings.cost_input_usd_per_1m_tokens,
        cached_input_usd_per_1m_tokens=settings.cost_cached_input_usd_per_1m_tokens,
        output_usd_per_1m_tokens=settings.cost_output_usd_per_1m_tokens,
    )


def _cost_breakdown_for_rows(rows, settings: Settings) -> dict[str, object]:
    return usage_cost_breakdown(
        [row.usage for row in rows], _cost_rates_from_settings(settings)
    )


def _require_questions(
    questions: list[GPQAQuestion], *, input_path: Path, domain: str
) -> None:
    if not questions:
        raise SystemExit(
            f"No {domain!r} questions found in {input_path}. "
            "Refusing to write a completed zero-row experiment artifact."
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _trace_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.trace.jsonl")


def _log_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.log")


def _manifest_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.manifest.json")


def _manifest_argv(args: argparse.Namespace) -> list[str]:
    return list(getattr(args, "_gpqa_argv", []))


@contextmanager
def _file_logging(path: Path, level_name: str):
    handler = _setup_file_logging(path, level_name)
    try:
        yield
    finally:
        root = logging.getLogger()
        root.removeHandler(handler)
        handler.close()


def _setup_file_logging(path: Path, level_name: str) -> logging.FileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, level_name.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(min(root.level or level, level))
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    )
    root.addHandler(handler)
    return handler


def _settings_manifest(settings: Settings) -> dict[str, object]:
    return {
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "git": _git_manifest(),
        },
        "llm": {
            "provider": settings.llm_provider,
            "main_model": settings.main_model,
            "subagent_model": settings.subagent_model,
            "self_consistency_model": settings.self_consistency_model,
            "reasoning_effort": settings.reasoning_effort,
            "max_output_tokens": settings.max_output_tokens,
            "json_max_retries": settings.json_max_retries,
        },
        "cost": {
            "cost_input_usd_per_1m_tokens": settings.cost_input_usd_per_1m_tokens,
            "cost_cached_input_usd_per_1m_tokens": (
                settings.cost_cached_input_usd_per_1m_tokens
            ),
            "cost_output_usd_per_1m_tokens": settings.cost_output_usd_per_1m_tokens,
            "max_total_api_calls": settings.max_total_api_calls,
            "max_total_cost_usd": settings.max_total_cost_usd,
        },
        "metrics": {
            "lambda_token": settings.lambda_token,
            "lambda_call": settings.lambda_call,
        },
        "logging": {"log_level": settings.log_level},
    }


def _git_manifest() -> dict[str, object]:
    commit = _git_output("rev-parse", "HEAD")
    branch = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    status = _git_output("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


# Mapping (CLI namespace attr) → env var. ``_apply_cli_overrides`` flips env
# values BEFORE ``get_settings()`` / ``make_client()`` consume them so every
# config has exactly one source of truth (env) plus a uniform CLI override
# path. Add an entry here when adding any new CLI flag that mirrors an env.
_CLI_TO_ENV: dict[str, str] = {
    "main_model": "MAIN_MODEL",
    "subagent_model": "SUBAGENT_MODEL",
    "self_consistency_model": "SELF_CONSISTENCY_MODEL",
    "reasoning_effort": "REASONING_EFFORT",
    "max_output_tokens": "MAX_OUTPUT_TOKENS",
    "json_max_retries": "LLM_JSON_MAX_RETRIES",
    "lambda_token": "LAMBDA_TOKEN",
    "lambda_call": "LAMBDA_CALL",
    "cost_input_usd_per_1m_tokens": "COST_INPUT_USD_PER_1M_TOKENS",
    "cost_cached_input_usd_per_1m_tokens": "COST_CACHED_INPUT_USD_PER_1M_TOKENS",
    "cost_output_usd_per_1m_tokens": "COST_OUTPUT_USD_PER_1M_TOKENS",
}


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    """Promote CLI-provided settings into ``os.environ`` for this process.

    Implements the project-wide convention that every configuration knob is
    available both as an env var AND a CLI flag: when the flag is provided
    we write through to ``os.environ`` and invalidate the cached settings
    snapshot, so the LLM client constructor and ``get_settings()`` callers
    see the override without per-call-site plumbing.
    """
    import os

    changed = False
    for attr, env_name in _CLI_TO_ENV.items():
        value = getattr(args, attr, None)
        if value is None:
            continue
        os.environ[env_name] = str(value)
        changed = True
    if changed:
        clear_settings_cache()


def _build_cost_guard(args: argparse.Namespace, settings: Settings) -> CostGuard:
    """Build a ``CostGuard`` from CLI flags + env-derived ``Settings``.

    Per dimension the tighter of (CLI flag, env default) wins so a forgotten
    ``--max-api-calls`` cannot silently lift a stricter ``MAX_TOTAL_API_CALLS``
    from the environment.
    """
    cli_calls = getattr(args, "max_api_calls", None)
    cli_cost = getattr(args, "max_estimated_cost_usd", None)
    env_calls = settings.max_total_api_calls
    env_cost = settings.max_total_cost_usd
    return CostGuard(
        max_api_calls=_tightest(cli_calls, env_calls),
        max_estimated_cost_usd=_tightest(cli_cost, env_cost),
        cost_input_usd_per_1m_tokens=settings.cost_input_usd_per_1m_tokens,
        cost_cached_input_usd_per_1m_tokens=settings.cost_cached_input_usd_per_1m_tokens,
        cost_output_usd_per_1m_tokens=settings.cost_output_usd_per_1m_tokens,
    )


def _tightest(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


# Provider aliases. Any name that maps to "openai_compatible" routes to the
# OpenAI-API-compatible client, which works against any vendor exposing that
# schema (OpenAI, Together, Groq, OpenRouter, Anyscale, Fireworks, DeepSeek,
# xAI, Mistral, local vLLM, local Ollama, etc.). Configure the endpoint via
# OPENAI_BASE_URL or LLM_BASE_URL. See docs/providers.md.
_OPENAI_COMPATIBLE_ALIASES = {
    "openai",
    "openai_compatible",
    "openai-compatible",
    "compatible",
    "vllm",
    "ollama",
    "together",
    "togetherai",
    "groq",
    "openrouter",
    "anyscale",
    "fireworks",
    "deepseek",
    "xai",
    "mistral",
    "perplexity",
    "lmstudio",
    "local",
}


def make_client(provider: str) -> LLMClient:
    name = provider.strip().lower()
    if name == "mock":
        return MockLLMClient()
    if name in {"azure_openai", "azure-openai", "azure"}:
        return AzureOpenAIClient()
    if name in _OPENAI_COMPATIBLE_ALIASES:
        return OpenAICompatibleClient()
    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider!r}. Use 'mock', 'azure_openai', "
        "or any OpenAI-API-compatible alias (see docs/providers.md)."
    )


if __name__ == "__main__":
    main()
