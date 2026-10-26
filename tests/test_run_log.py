from video_report.evaluation.evaluate import evaluate_run
from video_report.pipeline import create_run, execute_run


def test_logs_redact_credentials_and_evaluation_preserves_execution_log(
    tmp_path, make_bench, make_experiment, monkeypatch
):
    secret = "AQ.local-test-secret-should-never-appear"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    bench = make_bench({"r": ["f"]}, raw_by_call={"r": {"r::whole": secret}})
    cfg = make_experiment(bench)
    run = create_run(cfg, tmp_path / "runs", run_id="r", environ={})
    execute_run(run, mode="create")
    log = run.path / "smoke_test.log"
    before = log.read_bytes()
    assert secret not in before.decode()
    assert "[REDACTED]" in before.decode()
    assert "TOTAL USAGE / ESTIMATED COST" in before.decode()
    out, _ = evaluate_run(run.path, bench / "ground_truth.jsonl")
    evaluated = (out / "smoke_test.log").read_text()
    assert log.read_bytes() == before
    assert "invalid_output" in evaluated
    assert secret not in evaluated
    assert "cost_status" in evaluated and "unknown" in evaluated
