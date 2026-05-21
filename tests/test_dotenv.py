from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpqa_cmab.config import (
    _parse_dotenv_line,
    clear_settings_cache,
    get_settings,
    load_dotenv,
)


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    # Snapshot of relevant keys to isolate tests.
    for key in (
        "LLM_PROVIDER",
        "MAIN_MODEL",
        "SUBAGENT_MODEL",
        "LAMBDA_TOKEN",
        "DOTENV_TEST_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_parse_dotenv_line_basics():
    assert _parse_dotenv_line("FOO=bar") == ("FOO", "bar")
    assert _parse_dotenv_line('FOO="hello world"') == ("FOO", "hello world")
    assert _parse_dotenv_line("FOO='quoted'") == ("FOO", "quoted")
    assert _parse_dotenv_line("export FOO=bar") == ("FOO", "bar")
    assert _parse_dotenv_line("FOO=bar  # comment") == ("FOO", "bar")
    assert _parse_dotenv_line("# pure comment") is None
    assert _parse_dotenv_line("") is None
    assert _parse_dotenv_line("not-a-valid-line") is None


def test_load_dotenv_explicit_path(tmp_path: Path, monkeypatch):
    env = tmp_path / "custom.env"
    env.write_text("LLM_PROVIDER=mock\nMAIN_MODEL=from-dotenv\n", encoding="utf-8")
    loaded = load_dotenv(env)
    assert loaded == env
    assert os.environ["MAIN_MODEL"] == "from-dotenv"


def test_load_dotenv_does_not_override_existing(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DOTENV_TEST_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("DOTENV_TEST_KEY", "from-shell")
    load_dotenv(env)
    assert os.environ["DOTENV_TEST_KEY"] == "from-shell"


def test_load_dotenv_override_flag(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DOTENV_TEST_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("DOTENV_TEST_KEY", "from-shell")
    load_dotenv(env, override=True)
    assert os.environ["DOTENV_TEST_KEY"] == "from-file"


def test_load_dotenv_walks_up(tmp_path: Path, monkeypatch):
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (tmp_path / ".env").write_text("DOTENV_TEST_KEY=walked-up\n", encoding="utf-8")
    monkeypatch.chdir(nested)
    loaded = load_dotenv()
    assert loaded == tmp_path / ".env"
    assert os.environ["DOTENV_TEST_KEY"] == "walked-up"


def test_get_settings_consumes_dotenv(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=mock\nMAIN_MODEL=via-dotenv\nLAMBDA_TOKEN=0.25\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    settings = get_settings()
    assert settings.main_model == "via-dotenv"
    assert settings.lambda_token == 0.25


def test_cli_env_file_flag(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    env = tmp_path / "my.env"
    env.write_text("LLM_PROVIDER=mock\nMAIN_MODEL=cli-dotenv-model\n", encoding="utf-8")
    # Pre-set to ensure --env-file overrides existing env vars.
    monkeypatch.setenv("MAIN_MODEL", "from-shell")

    from gpqa_cmab.cli import main as cli_main

    # Use validate-data which only needs a dataset path.
    sample = tmp_path / "sample.jsonl"
    sample.write_text(
        '{"question_id":"q1","domain":"physics","question":"?",'
        '"choices":{"A":"a","B":"b","C":"c","D":"d"},"correct_answer":"A"}\n',
        encoding="utf-8",
    )
    cli_main(
        [
            "--env-file",
            str(env),
            "validate-data",
            "--input",
            str(sample),
        ]
    )
    assert os.environ["MAIN_MODEL"] == "cli-dotenv-model"
    assert get_settings().main_model == "cli-dotenv-model"


def test_cli_env_file_missing(tmp_path: Path):
    from gpqa_cmab.cli import main as cli_main

    with pytest.raises(SystemExit):
        cli_main(
            [
                "--env-file",
                str(tmp_path / "nope.env"),
                "validate-data",
                "--input",
                str(tmp_path),
            ]
        )
