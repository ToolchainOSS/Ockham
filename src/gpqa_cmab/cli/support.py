"""Shared CLI helpers: provider/client construction, cost-guard wiring,
run manifests, file logging, and the env-override convention.

These are private building blocks consumed by the per-command modules in
``gpqa_cmab.cli.commands_*``. Nothing here issues user-facing output beyond
the structured progress/preflight warnings streamed to ``stderr``.
"""

from __future__ import annotations

import logging
import platform
import random
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from gpqa_cmab.config import Settings, clear_settings_cache, get_settings
from gpqa_cmab.cost_guard import CostGuard, CostRates, usage_cost_breakdown
from gpqa_cmab.dataset import load_questions
from gpqa_cmab.llm.base import LLMClient
from gpqa_cmab.llm.mock import MockLLMClient
from gpqa_cmab.llm.openai_compatible import (
    AzureOpenAIClient,
    OpenAICompatibleClient,
)
from gpqa_cmab.schemas import CallTelemetry, GPQAQuestion

if TYPE_CHECKING:
    import argparse


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


def _pick_question(args: argparse.Namespace) -> GPQAQuestion:
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


def _progress(verbose: int, message: str) -> None:
    """Stream a progress line to stderr so users see activity in real time."""
    if verbose <= 0:
        return
    print(f"[quick-check] {message}", file=sys.stderr, flush=True)


def _cost_rates_from_settings(settings: Settings) -> CostRates:
    return CostRates(
        input_usd_per_1m_tokens=settings.cost_input_usd_per_1m_tokens,
        cached_input_usd_per_1m_tokens=settings.cost_cached_input_usd_per_1m_tokens,
        output_usd_per_1m_tokens=settings.cost_output_usd_per_1m_tokens,
    )


def _cost_breakdown_for_rows(
    rows: list[CallTelemetry], settings: Settings
) -> dict[str, object]:
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
def _file_logging(path: Path, level_name: str) -> Iterator[None]:
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
    except OSError, subprocess.TimeoutExpired:
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


def _tightest[N: (int, float)](a: N | None, b: N | None) -> N | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


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
    global _PREFLIGHT_WARNED  # noqa: PLW0603 — one-shot process-wide warning flag
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
