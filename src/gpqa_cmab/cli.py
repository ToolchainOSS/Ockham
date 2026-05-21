from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpqa_cmab.agents.subagents import run_all_subagents
from gpqa_cmab.config import get_settings
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
from gpqa_cmab.schemas import FactorialResult
from gpqa_cmab.telemetry import read_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpqa-cmab")
    sub = parser.add_subparsers(required=True)

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
    run_subagents.set_defaults(func=cmd_run_subagents)

    factorial = sub.add_parser("run-factorial")
    factorial.add_argument("--input", required=True, type=Path)
    factorial.add_argument("--domain", default="physics")
    factorial.add_argument("--subagent-cache", type=Path)
    factorial.add_argument("--output", required=True, type=Path)
    factorial.add_argument("--max-questions", type=int)
    factorial.add_argument("--max-api-calls", type=int)
    factorial.add_argument(
        "--max-estimated-cost-usd",
        type=float,
        help=(
            "Stop after a question once the cumulative estimated cost (using "
            "COST_USD_PER_1K_TOKENS) reaches this value."
        ),
    )
    factorial.add_argument(
        "--cost-usd-per-1k-tokens",
        type=float,
        default=None,
        help="Override COST_USD_PER_1K_TOKENS for cost estimation.",
    )
    factorial.add_argument("--dry-run", action="store_true")
    factorial.set_defaults(func=cmd_run_factorial)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--results", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    evaluate.set_defaults(func=cmd_evaluate)

    replay = sub.add_parser("replay-bandit")
    replay.add_argument("--results", required=True, type=Path)
    replay.add_argument(
        "--policy", choices=["superarm-ts", "structured-cmab"], default="superarm-ts"
    )
    replay.add_argument("--seeds", type=int, default=10)
    replay.add_argument("--output", required=True, type=Path)
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
    baselines.set_defaults(func=cmd_baselines)

    smoke = sub.add_parser("smoke-test")
    smoke.add_argument("--mock", action="store_true")
    smoke.set_defaults(func=cmd_smoke_test)
    return parser


def cmd_validate_data(args: argparse.Namespace) -> None:
    questions = load_questions(args.input, args.domain, args.max_questions)
    print(json.dumps({"domain": args.domain, "questions": len(questions)}))


def cmd_run_subagents(args: argparse.Namespace) -> None:
    questions = load_questions(args.input, args.domain, args.max_questions)
    settings = get_settings()
    client = make_client(settings.llm_provider)
    rows = []
    for question in questions:
        reports, telemetry_rows = run_all_subagents(
            client,
            question,
            experiment_id="subagent-cache",
            model=settings.subagent_model,
        )
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
    print(json.dumps({"cached_reports": len(rows), "output": str(args.output)}))


def cmd_run_factorial(args: argparse.Namespace) -> None:
    questions = load_questions(args.input, args.domain, args.max_questions)
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
    cost_rate = (
        args.cost_usd_per_1k_tokens
        if args.cost_usd_per_1k_tokens is not None
        else settings.cost_usd_per_1k_tokens
    )
    results = run_full_factorial(
        questions,
        make_client(settings.llm_provider),
        main_model=settings.main_model,
        subagent_model=settings.subagent_model,
        max_api_calls=args.max_api_calls,
        max_estimated_cost_usd=args.max_estimated_cost_usd,
        cost_usd_per_1k_tokens=cost_rate,
        subagent_cache=subagent_cache,
    )
    write_jsonl(args.output, results)
    print(
        json.dumps(
            {
                "rows": len(results),
                "output": str(args.output),
                "used_subagent_cache": subagent_cache is not None,
            }
        )
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    rows = [FactorialResult.model_validate(row) for row in read_jsonl(args.results)]
    write_evaluation_outputs(rows, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir)}))


def cmd_replay_bandit(args: argparse.Namespace) -> None:
    rows = [FactorialResult.model_validate(row) for row in read_jsonl(args.results)]
    steps = replay_bandit(rows, policy=args.policy, seeds=args.seeds)
    write_jsonl(args.output, steps)
    print(json.dumps({"steps": len(steps), "output": str(args.output)}))


def cmd_report(args: argparse.Namespace) -> None:
    write_report(args.results_dir, args.output)
    print(json.dumps({"output": str(args.output)}))


def cmd_run_self_consistency(args: argparse.Namespace) -> None:
    questions = load_questions(args.input, args.domain, args.max_questions)
    settings = get_settings()
    k_values = [int(value) for value in args.k_values.split(",") if value.strip()]
    rows = run_self_consistency_experiment(
        questions,
        make_client(settings.llm_provider),
        model=settings.self_consistency_model,
        k_values=k_values,
        seed=args.seed,
        temperature=args.temperature,
    )
    write_jsonl(args.output, rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


def cmd_baselines(args: argparse.Namespace) -> None:
    rows = [FactorialResult.model_validate(row) for row in read_jsonl(args.results)]
    summary = baseline_summary(
        rows,
        static_subset_id=args.static_subset,
        random_target_subset_id=args.target_subset,
        random_target_size=args.target_size,
        seeds=args.seeds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}))


def cmd_smoke_test(args: argparse.Namespace) -> None:
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
    client = MockLLMClient()
    questions = load_questions(sample, "physics")
    rows = run_full_factorial(
        questions,
        client,
        main_model="mock-main",
        subagent_model="mock-subagent",
        experiment_id="mock-smoke",
    )
    write_jsonl(results_path, rows)
    write_evaluation_outputs(rows, Path("artifacts/results"))
    steps = replay_bandit(rows, policy="superarm-ts", seeds=2)
    write_jsonl(Path("artifacts/results/bandit_replay_results.jsonl"), steps)
    write_jsonl(
        Path("artifacts/results/bootstrap_results.json"), [{"status": "available"}]
    )
    write_jsonl(
        Path("artifacts/results/bandit_summary.json"),
        [{"policy": "superarm-ts", "seeds": 2, "partial_information": True}],
    )
    sc_rows = run_self_consistency_experiment(
        questions,
        client,
        model="mock-self-consistency",
        k_values=[1, 4],
        seed=0,
    )
    write_jsonl(Path("artifacts/results/self_consistency_results.jsonl"), sc_rows)
    write_report(Path("artifacts/results"), Path("artifacts/reports/mvp_report.md"))
    print(
        json.dumps(
            {"ok": True, "factorial_rows": len(rows), "bandit_steps": len(steps)}
        )
    )


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
