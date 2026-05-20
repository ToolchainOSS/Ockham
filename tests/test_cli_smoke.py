from gpqa_cmab.cli import main


def test_cli_smoke_test_runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    main(["smoke-test", "--mock"])
    assert (tmp_path / "artifacts/results/full_factorial_results.jsonl").exists()
    assert (tmp_path / "artifacts/reports/mvp_report.md").exists()
