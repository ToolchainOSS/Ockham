"""Argument-parser construction for the ``gpqa-cmab`` CLI.

``build_parser`` wires every subcommand to its handler in the
``gpqa_cmab.cli.commands_*`` modules and the shared override-flag helpers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gpqa_cmab.cli.commands_core import (
    cmd_baselines,
    cmd_benchmark_cmab,
    cmd_evaluate,
    cmd_replay_bandit,
    cmd_report,
    cmd_run_factorial,
    cmd_run_self_consistency,
    cmd_run_subagents,
    cmd_validate_data,
)
from gpqa_cmab.cli.commands_gfn import cmd_train_gfn
from gpqa_cmab.cli.commands_quick import cmd_quick_check, cmd_smoke_test
from gpqa_cmab.telemetry_db_cli import register as _register_telemetry_db_cli


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

    def _add_llm_overrides(
        p: argparse.ArgumentParser, *, models: tuple[str, ...]
    ) -> None:
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

    def _add_cost_caps(p: argparse.ArgumentParser) -> None:
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

    def _add_lambdas(p: argparse.ArgumentParser) -> None:
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

    benchmark = sub.add_parser(
        "benchmark-cmab",
        help=(
            "Offline CMAB benchmark on the per-subset empirical accuracies "
            "saved in metrics_summary.json. Runs the fixed and legacy-buggy "
            "policies for many seeds against a Bernoulli environment so the "
            "bug-fix effect can be measured without re-spending LLM tokens."
        ),
    )
    benchmark.add_argument(
        "--metrics-summary",
        type=Path,
        default=None,
        help=(
            "Optional source of per-subset accuracy/cost aggregates. When "
            "omitted (the default), the canonical 86-question MVP "
            "aggregates baked into mvp_aggregates.py are used so the "
            "benchmark stays reproducible even if metrics_summary.json "
            "has been overwritten by a smoke-test."
        ),
    )
    benchmark.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/cmab_benchmark.json"),
        help="Where to write the comparison report.",
    )
    benchmark.add_argument("--seeds", type=int, default=200)
    benchmark.add_argument("--steps", type=int, default=86)
    _add_lambdas(benchmark)
    benchmark.set_defaults(func=cmd_benchmark_cmab)

    gfn = sub.add_parser(
        "train-gfn",
        help=(
            "Phase 1: train the CMAB-GFN subagent-subset explorer with the "
            "Trajectory Balance objective. Requires the [gfn] extra "
            "(``uv sync --extra gfn``)."
        ),
    )
    gfn.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/gfn"),
        help="Where to write the trained-policy artifacts and run manifest.",
    )
    gfn.add_argument(
        "--num-iters",
        type=int,
        default=2000,
        help="Number of TB optimiser steps (default: 2000).",
    )
    gfn.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Trajectories per TB-loss batch (default: 64).",
    )
    gfn.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam LR for the policy network (default: 1e-3).",
    )
    gfn.add_argument(
        "--log-z-learning-rate",
        type=float,
        default=1e-2,
        help="Adam LR for the scalar log Z (default: 1e-2).",
    )
    gfn.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Policy MLP hidden width (default: 64).",
    )
    gfn.add_argument(
        "--temperature",
        type=float,
        default=0.02,
        help=(
            "Reward sharpening temperature; R(x) = exp(utility / T). Smaller "
            "values concentrate the distribution on high-utility subsets. "
            "Default 0.02 is the value at which expected utility under the "
            "trained policy exceeds the static[C] baseline on the canonical "
            "86-question MVP factorial while preserving multi-mode diversity. "
            "Set 0.1 for the original Phase-1 prototype behaviour."
        ),
    )
    gfn.add_argument(
        "--cmab-filter",
        choices=("single-arm", "marginal", "none"),
        default="single-arm",
        help=(
            "CMAB pre-filter family. 'single-arm' uses each solo subset's "
            "utility; 'marginal' uses E[U|i in S] - E[U|i not in S]; 'none' "
            "disables the filter (pure GFN ablation)."
        ),
    )
    gfn.add_argument(
        "--gamma",
        type=float,
        default=0.6,
        help=(
            "Pruning threshold; arms with score < gamma are dropped from "
            "the GFN's action space (default: 0.6, which prunes B and D on "
            "the MVP empirical data when using --cmab-filter=single-arm)."
        ),
    )
    gfn.add_argument(
        "--eval-samples",
        type=int,
        default=1000,
        help="Number of trajectories to draw for post-training evaluation.",
    )
    gfn.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch manual seed.",
    )
    gfn.set_defaults(func=cmd_train_gfn)

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

    _register_telemetry_db_cli(sub)
    return parser
