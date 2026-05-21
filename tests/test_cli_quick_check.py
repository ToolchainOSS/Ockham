from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpqa_cmab.cli import main
from gpqa_cmab.config import clear_settings_cache


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_quick_check_full_subset(sample_jsonl: Path, capsys, monkeypatch):
    monkeypatch.chdir(sample_jsonl.parent)
    main(["quick-check", "--input", str(sample_jsonl), "--seed", "0"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["provider"] == "mock"
    assert payload["subset"] == "ABCD"
    assert payload["api_calls"] == 5
    assert payload["predicted_answer"] in {"A", "B", "C", "D"}
    assert payload["question_id"] == "q1"  # only physics question in fixture
    assert payload["estimated_cost_usd"] == 0.0


def test_quick_check_minimal_subset_is_cheapest(sample_jsonl: Path, capsys):
    main(
        [
            "quick-check",
            "--input",
            str(sample_jsonl),
            "--subset",
            "A",
            "--seed",
            "1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    # 1 subagent + 1 main integrator = 2 LLM calls. The cheapest sanity check.
    assert payload["api_calls"] == 2
    assert payload["subset"] == "A"


def test_quick_check_forces_mock_without_allow_flag(
    sample_jsonl: Path, capsys, monkeypatch
):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    clear_settings_cache()
    main(["quick-check", "--input", str(sample_jsonl), "--seed", "0"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "mock"
    assert payload["forced_mock"] is True


def test_quick_check_rejects_bad_subset(sample_jsonl: Path):
    with pytest.raises(SystemExit):
        main(
            [
                "quick-check",
                "--input",
                str(sample_jsonl),
                "--subset",
                "XZ",
            ]
        )
