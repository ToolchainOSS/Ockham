from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpqa_cmab.agents.subagents import run_all_subagents
from gpqa_cmab.config import get_settings
from gpqa_cmab.dataset import load_questions
from gpqa_cmab.experiments.factorial import run_full_factorial
from gpqa_cmab.experiments.replay import replay_bandit
from gpqa_cmab.llm.mock import MockLLMClient
from gpqa_cmab.llm.openai_client import OpenAIClient
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
    results = run_full_factorial(
        questions,
        make_client(settings.llm_provider),
        main_model=settings.main_model,
        subagent_model=settings.subagent_model,
        max_api_calls=args.max_api_calls,
    )
    write_jsonl(args.output, results)
    print(json.dumps({"rows": len(results), "output": str(args.output)}))


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
    rows = run_full_factorial(
        load_questions(sample, "physics"),
        MockLLMClient(),
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
    write_jsonl(
        Path("artifacts/results/self_consistency_results.jsonl"),
        [{"status": "available", "baselines": ["CoT-1", "SC-4", "SC-8", "SC-16"]}],
    )
    write_report(Path("artifacts/results"), Path("artifacts/reports/mvp_report.md"))
    print(
        json.dumps(
            {"ok": True, "factorial_rows": len(rows), "bandit_steps": len(steps)}
        )
    )


def make_client(provider: str):
    if provider == "mock":
        return MockLLMClient()
    if provider in {"openai", "azure_openai"}:
        return OpenAIClient()
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


if __name__ == "__main__":
    main()
