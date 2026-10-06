"""Runner: concurrency, retries, failure classes, checkpoints, interruption, resume, locking."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from conftest import ScriptedProvider, SimulatedCrash
from video_report.config import ExecutionConfig
from video_report.io_utils import sha256_file
from video_report.pipeline import (
    IncompatibleResumeError,
    check_resume_compatibility,
    create_run,
    execute_run,
)
from video_report.run_store import RunDir, RunError, RunLockedError
from video_report.runner import ExecutionPolicy, backoff_delay


def _factory(provider: ScriptedProvider) -> Callable[..., ScriptedProvider]:
    return lambda cfg, run: provider


@pytest.fixture
def setup(
    tmp_path: Path, make_bench: Callable[..., Path], make_experiment: Callable[..., Path]
) -> Callable[..., RunDir]:
    def _setup(
        method: str = "per_field", reports: dict[str, list[str]] | None = None, **cfg: Any
    ) -> RunDir:
        bench = make_bench(reports or {"r1": ["f1", "f2", "f3"], "r2": ["f1", "f2"]})
        config = make_experiment(bench, method=method, **cfg)
        return create_run(config, tmp_path / "runs", run_id="t", environ={})

    return _setup


def _calls(run: RunDir) -> dict[str, dict[str, Any]]:
    return {c["call_id"]: c for c in run.read_all_calls()}


def _predictions(run: RunDir) -> list[dict[str, Any]]:
    return [json.loads(line) for line in run.predictions_path.read_text().splitlines()]


def test_complete_run_and_consistency(setup: Callable[..., RunDir], no_sleep: Any) -> None:
    run = setup()
    provider = ScriptedProvider()
    res = execute_run(run, mode="create", provider_factory=_factory(provider), sleep=no_sleep)
    assert res.state == "completed" and res.counts == {"succeeded": 5}
    calls = _calls(run)
    preds = _predictions(run)
    assert sorted(p["call_id"] for p in preds) == sorted(calls)
    for p in preds:
        c = calls[p["call_id"]]
        assert c["final_attempt"] == p["attempt"] == len(c["attempts"])
        assert c["attempts"][-1]["response_file"] == p["response_file"]
        assert (run.path / p["response_file"]).is_file()
    meta = run.read_meta()
    assert meta["state"] == "completed" and len(meta["sessions"]) == 1


def test_out_of_order_completion_matched_by_id(setup: Callable[..., RunDir], no_sleep: Any) -> None:
    run = setup(execution={"concurrency": 5, "backoff_initial_s": 0.0})
    ids = [c.call_id for c in run.load_plan()]
    # Earlier calls are slower, so completion order is the reverse of submission order.
    delays = {cid: 0.05 * (len(ids) - i) for i, cid in enumerate(ids)}
    provider = ScriptedProvider(
        delays=delays, default=lambda r: json.dumps({r.field_ids[0]: r.call_id[-1].upper()})
    )
    execute_run(run, mode="create", provider_factory=_factory(provider), sleep=no_sleep)
    assert provider.completed != ids  # genuinely out of order
    for cid, state in _calls(run).items():
        payload = run.read_response(state["attempts"][-1]["response_file"])
        assert payload["call_id"] == cid
        assert json.loads(payload["response"]["text"]) == {state["field_ids"][0]: cid[-1].upper()}


def test_retryable_errors_retried_then_succeed(setup: Callable[..., RunDir], no_sleep: Any) -> None:
    run = setup(
        reports={"r1": ["f1"]},
        execution={"max_attempts": 3, "backoff_initial_s": 1.0, "backoff_jitter": 0.0},
    )
    provider = ScriptedProvider(script={"r1::field=f1": ["retryable", "timeout"]})
    res = execute_run(run, mode="create", provider_factory=_factory(provider), sleep=no_sleep)
    assert res.state == "completed"
    c = _calls(run)["r1::field=f1"]
    assert [a["outcome"] for a in c["attempts"]] == ["retryable_error", "timeout", "success"]
    assert c["attempts"][1]["error"]["outcome_unknown"] is True
    assert no_sleep.delays == [1.0, 2.0]  # exponential backoff, injected sleep


def test_retry_limit_exhausted(setup: Callable[..., RunDir], no_sleep: Any) -> None:
    run = setup(reports={"r1": ["f1"]}, execution={"max_attempts": 2, "backoff_initial_s": 0.0})
    provider = ScriptedProvider(script={"r1::field=f1": ["retryable"] * 5})
    res = execute_run(run, mode="create", provider_factory=_factory(provider), sleep=no_sleep)
    assert res.state == "completed_with_failures"
    c = _calls(run)["r1::field=f1"]
    assert c["status"] == "failed" and c["failure"] == "retries_exhausted"
    assert len(provider.calls) == 2 and len(c["attempts"]) == 2
    assert not _predictions(run)


def test_permanent_error_not_retried(setup: Callable[..., RunDir], no_sleep: Any) -> None:
    run = setup(reports={"r1": ["f1"]}, execution={"max_attempts": 5})
    provider = ScriptedProvider(script={"r1::field=f1": ["permanent", "retryable"]})
    execute_run(run, mode="create", provider_factory=_factory(provider), sleep=no_sleep)
    c = _calls(run)["r1::field=f1"]
    assert c["status"] == "failed" and c["failure"] == "permanent_error"
    assert len(provider.calls) == 1 and no_sleep.delays == []


def test_unexpected_adapter_exception_not_retried(
    setup: Callable[..., RunDir], no_sleep: Any
) -> None:
    run = setup(reports={"r1": ["f1"]})
    provider = ScriptedProvider(script={"r1::field=f1": [RuntimeError("boom key=SECRET123")]})
    execute_run(run, mode="create", provider_factory=_factory(provider), sleep=no_sleep)
    c = _calls(run)["r1::field=f1"]
    assert c["status"] == "failed" and c["failure"] == "unexpected_error"
    assert "SECRET123" not in json.dumps(c)
    assert len(provider.calls) == 1


def test_format_errors_are_not_retried(setup: Callable[..., RunDir], no_sleep: Any) -> None:
    run = setup(reports={"r1": ["f1"]})
    provider = ScriptedProvider(script={"r1::field=f1": ["this is not json"]})
    execute_run(run, mode="create", provider_factory=_factory(provider), sleep=no_sleep)
    c = _calls(run)["r1::field=f1"]
    assert c["status"] == "succeeded" and len(c["attempts"]) == 1
    assert run.read_response(c["attempts"][0]["response_file"])["response"]["text"] == (
        "this is not json"
    )


def test_backoff_formula() -> None:
    p = ExecutionPolicy.from_config(
        ExecutionConfig(backoff_initial_s=2, backoff_max_s=5, backoff_jitter=0.5)
    )
    rng = random.Random(0)
    d = [backoff_delay(i, p, rng) for i in (1, 2, 3, 4)]
    assert 2 <= d[0] <= 3 and 4 <= d[1] <= 6 and 5 <= d[2] <= 7.5 and 5 <= d[3] <= 7.5


def test_policy_overrides_validated() -> None:
    with pytest.raises(ValueError):
        ExecutionPolicy.from_config(ExecutionConfig(), concurrency=0)
    assert ExecutionPolicy.from_config(ExecutionConfig(), concurrency=7).concurrency == 7


# --------------------------------------------------------------------------- interruption


def test_crash_then_resume_does_not_repeat_successes(
    setup: Callable[..., RunDir], no_sleep: Any
) -> None:
    run = setup(execution={"concurrency": 1, "backoff_initial_s": 0.0})
    ids = [c.call_id for c in run.load_plan()]
    crash_at = ids[2]
    p1 = ScriptedProvider(script={crash_at: [SimulatedCrash()]})
    with pytest.raises(SimulatedCrash):
        execute_run(run, mode="create", provider_factory=_factory(p1), sleep=no_sleep)

    calls = _calls(run)
    assert [calls[i]["status"] for i in ids[:2]] == ["succeeded", "succeeded"]
    assert calls[crash_at]["status"] == "in_progress"
    assert run.read_meta()["state"] == "running"  # crash leaves no clean final state
    before = {i: sha256_file(run.call_path(i)) for i in ids[:2]}

    p2 = ScriptedProvider()
    res = execute_run(run, mode="resume", provider_factory=_factory(p2), sleep=no_sleep)
    assert res.state == "completed"
    assert res.recovered == {"from_response_file": 0, "marked_interrupted": 1}
    assert sorted(p2.calls) == sorted(ids[2:])  # successes not re-requested
    assert {i: sha256_file(run.call_path(i)) for i in ids[:2]} == before
    crashed = _calls(run)[crash_at]
    assert [a["outcome"] for a in crashed["attempts"]] == ["interrupted", "success"]
    assert crashed["attempts"][0]["error"]["outcome_unknown"] is True
    assert crashed["final_attempt"] == 2
    preds = _predictions(run)
    assert len(preds) == len(ids) == len({p["call_id"] for p in preds})  # no duplicates
    assert [s["mode"] for s in run.read_meta()["sessions"]] == ["create", "resume"]


def test_crash_after_response_saved_is_recovered_without_new_request(
    setup: Callable[..., RunDir], no_sleep: Any
) -> None:
    run = setup(reports={"r1": ["f1"]})
    cid = "r1::field=f1"
    # Simulate: attempt started, response persisted, crash before checkpoint update.
    state = run.read_call(cid)
    state["attempts"].append({"attempt": 1, "session": 1, "outcome": "in_progress"})
    state["status"] = "in_progress"
    run.write_call(state)
    run.write_response(
        cid,
        1,
        {
            "call_id": cid,
            "attempt": 1,
            "received_at": "t",
            "response": {"text": '{"f1": "B"}', "usage": None},
        },
    )
    p = ScriptedProvider()
    res = execute_run(run, mode="resume", provider_factory=_factory(p), sleep=no_sleep)
    assert res.recovered["from_response_file"] == 1 and p.calls == []
    c = _calls(run)[cid]
    assert c["status"] == "succeeded" and c["attempts"][0]["recovered"] is True
    assert _predictions(run)[0]["text"] == '{"f1": "B"}'


def test_keyboard_interrupt_is_graceful_and_resumable(
    setup: Callable[..., RunDir], no_sleep: Any
) -> None:
    run = setup(execution={"concurrency": 1, "backoff_initial_s": 0.0})
    ids = [c.call_id for c in run.load_plan()]
    p1 = ScriptedProvider(script={ids[1]: [KeyboardInterrupt()]})
    res = execute_run(run, mode="create", provider_factory=_factory(p1), sleep=no_sleep)
    assert res.state == "interrupted"
    assert run.read_meta()["state"] == "interrupted"
    res2 = execute_run(
        run, mode="resume", provider_factory=_factory(ScriptedProvider()), sleep=no_sleep
    )
    assert res2.state == "completed"


def test_failed_calls_only_rerun_with_retry_failed(
    setup: Callable[..., RunDir], no_sleep: Any
) -> None:
    run = setup(reports={"r1": ["f1", "f2"]})
    p1 = ScriptedProvider(script={"r1::field=f1": ["permanent"]})
    assert (
        execute_run(run, mode="create", provider_factory=_factory(p1), sleep=no_sleep).state
        == "completed_with_failures"
    )
    p2 = ScriptedProvider()
    res = execute_run(run, mode="resume", provider_factory=_factory(p2), sleep=no_sleep)
    assert p2.calls == [] and res.state == "completed_with_failures"
    p3 = ScriptedProvider()
    res = execute_run(
        run, mode="resume", retry_failed=True, provider_factory=_factory(p3), sleep=no_sleep
    )
    assert p3.calls == ["r1::field=f1"] and res.state == "completed"
    c = _calls(run)["r1::field=f1"]
    assert [a["outcome"] for a in c["attempts"]] == ["permanent_error", "success"]
    assert [a["session"] for a in c["attempts"]] == [1, 3]


# --------------------------------------------------------------------------- lock


def test_second_writer_rejected(setup: Callable[..., RunDir], no_sleep: Any) -> None:
    run = setup()
    other = RunDir(run.path)
    with run.writer_lock():
        assert other.lock_is_held()
        with pytest.raises(RunLockedError):
            execute_run(
                other, mode="resume", provider_factory=_factory(ScriptedProvider()), sleep=no_sleep
            )
    assert not other.lock_is_held()
    assert all(c["status"] == "pending" for c in other.read_all_calls())


def test_meta_write_requires_lock(setup: Callable[..., RunDir]) -> None:
    run = setup()
    with pytest.raises(RunError, match="without holding"):
        run.write_meta(run.read_meta())


def test_run_directory_is_exclusive(
    tmp_path: Path, make_bench: Callable[..., Path], make_experiment: Callable[..., Path]
) -> None:
    cfg = make_experiment(make_bench({"r1": ["f1"]}))
    create_run(cfg, tmp_path / "runs", run_id="same", environ={})
    with pytest.raises(RunError, match="already exists"):
        create_run(cfg, tmp_path / "runs", run_id="same", environ={})


# --------------------------------------------------------------------------- snapshots


def test_source_edits_do_not_change_frozen_requests(
    tmp_path: Path,
    make_bench: Callable[..., Path],
    make_experiment: Callable[..., Path],
    no_sleep: Any,
) -> None:
    bench = make_bench({"r1": ["f1"]})
    cfg = make_experiment(bench, user_prompt="ORIGINAL PROMPT\n$questions\n")
    run = create_run(cfg, tmp_path / "runs", run_id="t", environ={})
    (cfg.parent / "prompts" / "user.txt").write_text("EDITED PROMPT\n$questions\n")
    reports = (bench / "reports.jsonl").read_text().replace("Question about f1?", "CHANGED?")
    (bench / "reports.jsonl").write_text(reports)
    provider = ScriptedProvider()
    execute_run(run, mode="create", provider_factory=_factory(provider), sleep=no_sleep)
    assert provider.requests[0].user.startswith("ORIGINAL PROMPT")
    assert "Question about f1?" in provider.requests[0].user
    # Resuming against the edited sources is rejected; resuming from the snapshot is fine.
    with pytest.raises(IncompatibleResumeError, match="prompts"):
        check_resume_compatibility(
            run, config_path=cfg, video_root=None, allow_code_change=True, environ={}
        )
    check_resume_compatibility(run, config_path=None, video_root=None, allow_code_change=True)


def test_resume_rejects_changed_settings(
    tmp_path: Path, make_bench: Callable[..., Path], make_experiment: Callable[..., Path]
) -> None:
    bench = make_bench({"r1": ["f1"]})
    cfg = make_experiment(bench)
    run = create_run(cfg, tmp_path / "runs", run_id="t", environ={})
    check_resume_compatibility(
        run, config_path=cfg, video_root=None, allow_code_change=True, environ={}
    )
    text = cfg.read_text().replace("mock-test", "mock-other-model")
    cfg.write_text(text)
    with pytest.raises(IncompatibleResumeError, match="config"):
        check_resume_compatibility(
            run, config_path=cfg, video_root=None, allow_code_change=True, environ={}
        )


def test_execution_settings_may_change_on_resume(
    tmp_path: Path, make_bench: Callable[..., Path], make_experiment: Callable[..., Path]
) -> None:
    bench = make_bench({"r1": ["f1"]})
    cfg = make_experiment(bench)
    run = create_run(cfg, tmp_path / "runs", run_id="t", environ={})
    cfg.write_text(cfg.read_text().replace("concurrency: 2", "concurrency: 9"))
    check_resume_compatibility(
        run, config_path=cfg, video_root=None, allow_code_change=True, environ={}
    )


def test_tampered_snapshot_rejected(setup: Callable[..., RunDir]) -> None:
    run = setup()
    p = run.snapshot / "requests.jsonl"
    p.write_text(p.read_text().replace("Question about", "Tampered"))
    with pytest.raises(RunError, match="requests"):
        run.verify_snapshot()


def test_code_change_detected(
    setup: Callable[..., RunDir], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = setup()
    import video_report.pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "code_revision",
        lambda: {"commit": "0" * 40, "dirty": False, "diff_sha256": None, "source_sha256": "f"},
    )
    with pytest.raises(IncompatibleResumeError, match="package source changed"):
        check_resume_compatibility(run, config_path=None, video_root=None, allow_code_change=False)
    notes = check_resume_compatibility(
        run, config_path=None, video_root=None, allow_code_change=True
    )
    assert "allowed" in notes[0]


def test_provider_description_change_rejected(setup: Callable[..., RunDir], no_sleep: Any) -> None:
    run = setup(reports={"r1": ["f1"]})
    p = ScriptedProvider(script={"r1::field=f1": [KeyboardInterrupt()]})
    execute_run(run, mode="create", provider_factory=_factory(p), sleep=no_sleep)
    p2 = ScriptedProvider()
    p2.model = "different-model"
    with pytest.raises(IncompatibleResumeError, match="provider description changed"):
        execute_run(run, mode="resume", provider_factory=_factory(p2), sleep=no_sleep)


def test_drain_survives_repeated_interrupts() -> None:
    from video_report.runner import drain

    class Pool:
        calls = 0

        def shutdown(self, wait: bool, cancel_futures: bool) -> None:
            Pool.calls += 1
            if Pool.calls < 3:
                raise KeyboardInterrupt  # a second and third Ctrl-C while waiting

    drain(Pool())  # type: ignore[arg-type]
    assert Pool.calls == 3


def test_executed_calls_counts_only_attempted(setup: Callable[..., RunDir], no_sleep: Any) -> None:
    run = setup(execution={"concurrency": 1, "backoff_initial_s": 0.0})
    ids = [c.call_id for c in run.load_plan()]
    p = ScriptedProvider(script={ids[1]: [KeyboardInterrupt()]})
    res = execute_run(run, mode="create", provider_factory=_factory(p), sleep=no_sleep)
    assert res.state == "interrupted" and res.executed_calls == 2 < len(ids)


def test_untracked_source_edit_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from video_report import snapshot

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(snapshot, "__file__", str(pkg / "snapshot.py"))
    before = snapshot.package_source_sha256()
    (pkg / "new_untracked.py").write_text("y = 2\n")
    assert snapshot.package_source_sha256() != before


def test_unknown_source_hash_rejected(
    setup: Callable[..., RunDir], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = setup()
    env = run.load_environment()
    env["code"].pop("source_sha256")
    (run.snapshot / "environment.json").write_text(json.dumps(env))
    with pytest.raises(IncompatibleResumeError, match="changed or unknown"):
        check_resume_compatibility(run, config_path=None, video_root=None, allow_code_change=False)


def test_same_source_different_commit_is_a_note(
    setup: Callable[..., RunDir], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = setup()
    import video_report.pipeline as pipeline

    src = run.load_environment()["code"]["source_sha256"]
    monkeypatch.setattr(
        pipeline,
        "code_revision",
        lambda: {"commit": "1" * 40, "dirty": False, "diff_sha256": None, "source_sha256": src},
    )
    notes = check_resume_compatibility(
        run, config_path=None, video_root=None, allow_code_change=False
    )
    assert "package source is identical" in notes[0]


def test_probe_does_not_block_or_create_lock(setup: Callable[..., RunDir]) -> None:
    import fcntl
    import os
    import threading

    run = setup()
    assert not run.lock_path.exists()
    assert run.lock_is_held() is False
    assert not run.lock_path.exists()  # probing never creates the lock file
    run.lock_path.touch()
    # Hold a probe-style shared lock briefly; a starting writer must wait, not fail.
    fd = os.open(run.lock_path, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_SH)
    threading.Timer(0.2, lambda: (fcntl.flock(fd, fcntl.LOCK_UN), os.close(fd))).start()
    with run.writer_lock():
        assert RunDir(run.path).lock_is_held()
