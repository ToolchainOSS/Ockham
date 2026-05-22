from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from gpqa_cmab.cli import main
from gpqa_cmab.config import clear_settings_cache


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("COST_INPUT_USD_PER_1M_TOKENS", "0")
    monkeypatch.setenv("COST_CACHED_INPUT_USD_PER_1M_TOKENS", "0")
    monkeypatch.setenv("COST_OUTPUT_USD_PER_1M_TOKENS", "0")
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_quick_check_full_factorial_default(sample_jsonl: Path, capsys, monkeypatch):
    monkeypatch.chdir(sample_jsonl.parent)
    main(["quick-check", "--input", str(sample_jsonl), "--seed", "0"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["mode"] == "factorial"
    assert payload["provider"] == "mock"
    assert payload["subsets_evaluated"] == 16
    # 4 subagents + 16 main integrator calls.
    assert payload["api_calls"] == 20
    assert payload["question_id"] == "q1"  # only physics question in fixture
    assert payload["estimated_cost_usd"] == 0.0
    assert payload["full_subset_predicted"] in {"A", "B", "C", "D"}
    assert isinstance(payload["per_subset"], list) and len(payload["per_subset"]) == 16
    # Artifacts written to the default output dir.
    output_dir = Path(payload["output_dir"])
    assert (output_dir / "full_factorial_results.jsonl").exists()
    assert Path(payload["trace"]).exists()
    assert Path(payload["manifest"]).exists()
    assert (output_dir / "quick_check_summary.json").exists()
    trace_rows = [
        json.loads(line)
        for line in Path(payload["trace"]).read_text(encoding="utf-8").splitlines()
    ]
    assert len(trace_rows) == 20
    assert all(row["prompt_sha256"] for row in trace_rows)
    manifest = json.loads(Path(payload["manifest"]).read_text(encoding="utf-8"))
    assert manifest["command"] == "quick-check"
    assert manifest["artifacts"]
    assert manifest["argv"][:2] == ["gpqa-cmab", "quick-check"]
    assert manifest["traces"][0]["path"] == payload["trace"]
    assert manifest["trace_summary"]["call_rows"] == 20
    assert Path(payload["log"]).exists()


def test_quick_check_minimal_subset_is_cheapest(
    sample_jsonl: Path, capsys, monkeypatch
):
    monkeypatch.chdir(sample_jsonl.parent)
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
    assert payload["mode"] == "single-subset"
    assert payload["api_calls"] == 2
    assert payload["subset"] == "A"
    assert Path(payload["trace"]).exists()
    assert Path(payload["manifest"]).exists()


def test_quick_check_reports_tiered_cost_breakdown(
    sample_jsonl: Path, capsys, monkeypatch
):
    monkeypatch.chdir(sample_jsonl.parent)
    monkeypatch.setenv("COST_INPUT_USD_PER_1M_TOKENS", "1")
    monkeypatch.setenv("COST_CACHED_INPUT_USD_PER_1M_TOKENS", "0.1")
    monkeypatch.setenv("COST_OUTPUT_USD_PER_1M_TOKENS", "10")
    clear_settings_cache()

    main(["quick-check", "--input", str(sample_jsonl), "--subset", "A"])
    payload = json.loads(capsys.readouterr().out)
    tokens = payload["tokens"]
    expected = (tokens["prompt"] * 1 + tokens["completion"] * 10) / 1_000_000

    assert payload["cost_breakdown"]["pricing"]["mode"] == "tiered"
    assert payload["estimated_cost_usd"] == expected


def test_quick_check_forces_mock_without_allow_flag(
    sample_jsonl: Path, capsys, monkeypatch
):
    monkeypatch.chdir(sample_jsonl.parent)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    clear_settings_cache()
    main(["quick-check", "--input", str(sample_jsonl), "--seed", "0", "--subset", "A"])
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


def test_experiment_commands_reject_empty_filtered_dataset(
    sample_jsonl: Path, tmp_path: Path
):
    with pytest.raises(SystemExit, match="No 'astrophysics' questions"):
        main(
            [
                "run-subagents",
                "--input",
                str(sample_jsonl),
                "--domain",
                "astrophysics",
                "--output",
                str(tmp_path / "cache.jsonl"),
            ]
        )


def test_quick_check_file_logging_handler_is_scoped(
    sample_jsonl: Path, capsys, monkeypatch
):
    monkeypatch.chdir(sample_jsonl.parent)
    before = list(logging.getLogger().handlers)
    main(["quick-check", "--input", str(sample_jsonl), "--subset", "A"])
    capsys.readouterr()
    after = list(logging.getLogger().handlers)
    assert after == before
