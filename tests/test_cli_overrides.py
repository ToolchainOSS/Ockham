"""Locks the contract that every CLI override flag mirrors an env var.

The project convention is single-source-of-truth env vars + uniform CLI
overrides via ``_apply_cli_overrides`` (cli.py). If anyone adds an override
flag without wiring it into ``_CLI_TO_ENV`` (or vice-versa), these tests
fail loudly.
"""

from __future__ import annotations

import argparse
import os

import pytest

from gpqa_cmab.cli import _CLI_TO_ENV, _apply_cli_overrides
from gpqa_cmab.config import clear_settings_cache, get_settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    # Wipe any inherited override values so each test starts clean.
    for env_name in _CLI_TO_ENV.values():
        monkeypatch.delenv(env_name, raising=False)
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_apply_cli_overrides_writes_env_and_invalidates_cache():
    args = argparse.Namespace(
        main_model="cli-main",
        subagent_model="cli-sub",
        self_consistency_model="cli-sc",
        reasoning_effort="medium",
        max_output_tokens=512,
        json_max_retries=1,
        lambda_token=0.33,
        lambda_call=0.07,
        cost_input_usd_per_1m_tokens=1.25,
        cost_cached_input_usd_per_1m_tokens=0.125,
        cost_output_usd_per_1m_tokens=10.0,
    )
    _apply_cli_overrides(args)
    assert os.environ["MAIN_MODEL"] == "cli-main"
    assert os.environ["SUBAGENT_MODEL"] == "cli-sub"
    assert os.environ["SELF_CONSISTENCY_MODEL"] == "cli-sc"
    assert os.environ["REASONING_EFFORT"] == "medium"
    assert os.environ["MAX_OUTPUT_TOKENS"] == "512"
    assert os.environ["LLM_JSON_MAX_RETRIES"] == "1"
    settings = get_settings()
    assert settings.main_model == "cli-main"
    assert settings.subagent_model == "cli-sub"
    assert settings.self_consistency_model == "cli-sc"
    assert settings.reasoning_effort == "medium"
    assert settings.max_output_tokens == 512
    assert settings.json_max_retries == 1
    assert settings.lambda_token == 0.33
    assert settings.lambda_call == 0.07
    assert settings.cost_input_usd_per_1m_tokens == 1.25
    assert settings.cost_cached_input_usd_per_1m_tokens == 0.125
    assert settings.cost_output_usd_per_1m_tokens == 10.0


def test_apply_cli_overrides_is_noop_when_flags_unset(monkeypatch):
    monkeypatch.setenv("MAIN_MODEL", "env-main")
    clear_settings_cache()
    args = argparse.Namespace(**{attr: None for attr in _CLI_TO_ENV})
    _apply_cli_overrides(args)
    assert os.environ["MAIN_MODEL"] == "env-main"
    assert get_settings().main_model == "env-main"


def test_dotenv_example_documents_every_env_var():
    """All env names the code reads must be mentioned in .env.example.

    Catches drift between Settings/CostGuard/LLM client and .env.example.
    """
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    example_text = (repo_root / ".env.example").read_text(encoding="utf-8")

    # Env vars whose presence we require in .env.example. Keep this list
    # explicit so adding a new env var is a one-line change in two places.
    required = {
        "LOG_LEVEL",
        "LLM_PROVIDER",
        "MAIN_MODEL",
        "SUBAGENT_MODEL",
        "SELF_CONSISTENCY_MODEL",
        "REASONING_EFFORT",
        "LLM_USE_RESPONSES_API",
        "MAX_OUTPUT_TOKENS",
        "OPENAI_API_KEY",
        "OPENAI_API_KEYS",
        "LLM_API_KEY",
        "OPENAI_BASE_URL",
        "LLM_BASE_URL",
        "OPENAI_ORGANIZATION",
        "OPENAI_KEY_COOLDOWN_S",
        "OPENAI_MAX_RETRIES",
        "OPENAI_MAX_WAIT_S",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "LLM_DEFAULT_HEADERS",
        "LLM_TIMEOUT_S",
        "LAMBDA_TOKEN",
        "LAMBDA_CALL",
        "COST_INPUT_USD_PER_1M_TOKENS",
        "COST_CACHED_INPUT_USD_PER_1M_TOKENS",
        "COST_OUTPUT_USD_PER_1M_TOKENS",
        "MAX_TOTAL_API_CALLS",
        "MAX_TOTAL_COST_USD",
        "LLM_JSON_MAX_RETRIES",
    }
    documented = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", example_text, re.MULTILINE))
    missing = required - documented
    assert not missing, f"Missing from .env.example: {sorted(missing)}"
