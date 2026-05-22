import json

from gpqa_cmab.cli import main


def test_cli_smoke_test_runs(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    main(["smoke-test", "--mock"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "mock"
    assert payload["mock_flag"] is True
    assert (tmp_path / "artifacts/results/full_factorial_results.jsonl").exists()
    assert (tmp_path / "artifacts/reports/mvp_report.md").exists()
