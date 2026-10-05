"""Multi-process integration tests through the real CLI scripts (mock provider, no network)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from video_report.run_store import RunDir

REPO = Path(__file__).resolve().parents[1]
GT = REPO / "benchmarks/mock_mcq_v1/ground_truth.jsonl"


def cli(
    script: str, *args: str, env: dict[str, str] | None = None, background: bool = False
) -> Any:
    cmd = [sys.executable, str(REPO / "scripts" / script), *args]
    full_env = {**os.environ, **(env or {})}
    full_env.pop("VIDEO_REPORT_VIDEO_ROOT", None)
    if background:
        return subprocess.Popen(
            cmd, cwd=REPO, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    return subprocess.run(cmd, cwd=REPO, env=full_env, capture_output=True, text=True, timeout=120)


def copy_experiment(
    tmp_path: Path, exp: str, new_name: str, marker: str | None = None, **changes: Any
) -> Path:
    dst = tmp_path / new_name
    shutil.copytree(REPO / "experiments" / exp, dst)
    cfg = yaml.safe_load((dst / "config.yaml").read_text())
    cfg["name"] = new_name
    cfg["data"]["benchmark"] = str(REPO / "benchmarks/mock_mcq_v1")
    for key, value in changes.items():
        section, _, sub = key.partition("__")
        cfg[section][sub] = value
    (dst / "config.yaml").write_text(yaml.safe_dump(cfg))
    if marker:
        user = dst / "prompts/user.txt"
        user.write_text(f"{marker}\n" + user.read_text())
    return dst / "config.yaml"


def wait_for_session(run_path: Path, timeout: float = 30) -> None:
    """Wait until the writer has recorded its session (it holds the lock from then on)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            meta = json.loads((run_path / "run.json").read_text())
            if meta.get("sessions"):
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.05)
    raise AssertionError("writer never started")


def requests_of(run_path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (run_path / "snapshot/requests.jsonl").read_text().splitlines()]


def test_two_concurrent_cli_runs_do_not_contaminate(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    a = copy_experiment(
        tmp_path, "exp001_whole_mcq", "exp_a", "PROMPT-MARKER-A", provider__mock={"delay_s": 0.3}
    )
    b = copy_experiment(
        tmp_path,
        "exp002_per_field_mcq",
        "exp_b",
        "PROMPT-MARKER-B",
        provider__mock={"delay_s": 0.3},
    )
    pa = cli(
        "run_experiment.py",
        "run",
        "--config",
        str(a),
        "--runs-dir",
        str(runs),
        "--run-id",
        "A",
        background=True,
    )
    pb = cli(
        "run_experiment.py",
        "run",
        "--config",
        str(b),
        "--runs-dir",
        str(runs),
        "--run-id",
        "B",
        background=True,
    )
    outs = [p.communicate(timeout=120) for p in (pa, pb)]
    assert pa.returncode == 0 and pb.returncode == 0, outs

    ra, rb = requests_of(runs / "A"), requests_of(runs / "B")
    assert len(ra) == 1 and len(rb) == 3
    assert all("PROMPT-MARKER-A" in r["user"] and "MARKER-B" not in r["user"] for r in ra)
    assert all("PROMPT-MARKER-B" in r["user"] and "MARKER-A" not in r["user"] for r in rb)
    for rid, n in (("A", 1), ("B", 3)):
        preds = (runs / rid / "predictions.jsonl").read_text().splitlines()
        assert len(preds) == n
        calls = {c.call_id for c in RunDir(runs / rid).load_plan()}
        assert {json.loads(p)["call_id"] for p in preds} == calls
    for rid, acc in (("A", 2), ("B", 1)):
        r = cli("evaluate.py", "--run", str(runs / rid), "--ground-truth", str(GT))
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout)["field_accuracy"]["numerator"] == acc


def test_second_writer_rejected_and_mid_run_prompt_edit_ignored(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    cfg = copy_experiment(
        tmp_path,
        "exp002_per_field_mcq",
        "exp_lock",
        "ORIGINAL-MARKER",
        provider__mock={"delay_s": 0.5},
        execution__concurrency=1,
    )
    first = cli(
        "run_experiment.py",
        "run",
        "--config",
        str(cfg),
        "--runs-dir",
        str(runs),
        "--run-id",
        "L",
        background=True,
    )
    try:
        wait_for_session(runs / "L")
        # Edit the source prompt while the run is executing.
        user = cfg.parent / "prompts/user.txt"
        user.write_text(user.read_text().replace("ORIGINAL-MARKER", "EDITED-MARKER"))
        second = cli("run_experiment.py", "resume", str(runs / "L"), "--allow-code-change")
        assert second.returncode == 5, second.stdout + second.stderr
        assert "locked by another active writer" in second.stderr
    finally:
        out, err = first.communicate(timeout=120)
    assert first.returncode == 0, out + err
    reqs = requests_of(runs / "L")
    assert all("ORIGINAL-MARKER" in r["user"] and "EDITED" not in r["user"] for r in reqs)
    assert "ORIGINAL-MARKER" in (runs / "L/snapshot/prompts/user.txt").read_text()
    meta = json.loads((runs / "L/run.json").read_text())
    assert meta["state"] == "completed" and len(meta["sessions"]) == 1
    # Resume with the edited source config is rejected as incompatible.
    r = cli(
        "run_experiment.py", "resume", str(runs / "L"), "--config", str(cfg), "--allow-code-change"
    )
    assert r.returncode == 2 and "prompts" in r.stderr


def test_hard_crash_then_cli_resume(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    cfg = copy_experiment(tmp_path, "exp002_per_field_mcq", "exp_crash", execution__concurrency=1)
    crash_call = "synthetic-0001::field=first_item_in_cup"
    r = cli(
        "run_experiment.py",
        "run",
        "--config",
        str(cfg),
        "--runs-dir",
        str(runs),
        "--run-id",
        "C",
        env={"VIDEO_REPORT_MOCK_CRASH_ON_CALL": crash_call},
    )
    assert r.returncode == 75, r.stdout + r.stderr  # process died mid-call

    status = cli("run_experiment.py", "status", str(runs / "C"))
    st = json.loads(status.stdout)
    assert st["state"].startswith("running (stale") and st["active_writer"] is False
    assert st["calls"] == {"succeeded": 1, "in_progress": 1, "pending": 1}

    first_call = RunDir(runs / "C").read_call("synthetic-0001::field=primary_appliance")
    r = cli(
        "run_experiment.py", "resume", str(runs / "C"), "--config", str(cfg), "--allow-code-change"
    )
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads(r.stdout[r.stdout.index("{") :])
    assert out["recovered"] == {"from_response_file": 0, "marked_interrupted": 1}
    assert out["executed_calls"] == 2

    run = RunDir(runs / "C")
    assert run.read_call("synthetic-0001::field=primary_appliance") == first_call  # untouched
    crashed = run.read_call(crash_call)
    assert [a["outcome"] for a in crashed["attempts"]] == ["interrupted", "success"]
    preds = [json.loads(x) for x in run.predictions_path.read_text().splitlines()]
    assert sorted(p["call_id"] for p in preds) == sorted(c.call_id for c in run.load_plan())
    e = cli("evaluate.py", "--run", str(runs / "C"), "--ground-truth", str(GT))
    summary = json.loads(e.stdout)
    assert summary["provisional"] is False and summary["field_accuracy"]["numerator"] == 1


def test_mock_run_works_without_gemini_sdk(tmp_path: Path) -> None:
    code = (
        "import sys, importlib.abc\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name == 'google' or name.startswith('google.'):\n"
        "            raise ImportError('blocked for test: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from video_report.cli import main\n"
        "rc = main(sys.argv[1:])\n"
        "assert not any(m.startswith('google') for m in sys.modules), 'SDK imported'\n"
        "sys.exit(rc)\n"
    )
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "run",
            "--config",
            str(REPO / "experiments/exp001_whole_mcq/config.yaml"),
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            "nosdk",
            "--quiet",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        env={k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"},
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_gemini_config_without_model_or_videos_fails_before_any_call(tmp_path: Path) -> None:
    r = cli(
        "run_experiment.py",
        "run",
        "--config",
        str(REPO / "experiments/exp003_whole_gemini_smoke/config.yaml"),
        "--runs-dir",
        str(tmp_path),
    )
    assert r.returncode == 2 and "SET-ME" in r.stderr
    cfg = copy_experiment(tmp_path, "exp003_whole_gemini_smoke", "g", provider__model="m-x")
    r = cli("run_experiment.py", "run", "--config", str(cfg), "--runs-dir", str(tmp_path / "r"))
    assert r.returncode == 2 and "missing local video files" in r.stderr
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize("bad", ["--concurrency=0", "--max-attempts=0"])
def test_invalid_overrides_rejected(tmp_path: Path, bad: str) -> None:
    r = cli(
        "run_experiment.py",
        "run",
        "--config",
        str(REPO / "experiments/exp001_whole_mcq/config.yaml"),
        "--runs-dir",
        str(tmp_path),
        bad,
    )
    assert r.returncode == 2


def test_failed_preflight_leaves_no_run_directory(tmp_path: Path) -> None:
    r = cli(
        "run_experiment.py",
        "run",
        "--config",
        str(REPO / "experiments/exp001_whole_mcq/config.yaml"),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--concurrency=0",
    )
    assert r.returncode == 2 and not (tmp_path / "runs").exists()
    # Gemini without credentials: rejected before the run directory is written.
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    cfg = copy_experiment(tmp_path, "exp003_whole_gemini_smoke", "g2", provider__model="m-x")
    r = cli(
        "run_experiment.py",
        "run",
        "--config",
        str(cfg),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--video-root",
        str(tmp_path / "vids"),
        env={"GEMINI_API_KEY": ""},
    )
    assert r.returncode == 2
    assert "missing local video files" in r.stderr  # videos are checked first
    (tmp_path / "vids/videos").mkdir(parents=True)
    video.rename(tmp_path / "vids/videos/synthetic_kitchen_0001.mp4")
    r = cli(
        "run_experiment.py",
        "run",
        "--config",
        str(cfg),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--video-root",
        str(tmp_path / "vids"),
        env={"GEMINI_API_KEY": ""},
    )
    # Without the SDK installed, the SDK check fires first; with it, the missing key does.
    sdk = importlib.util.find_spec("google") and importlib.util.find_spec("google.genai")
    expected = "GEMINI_API_KEY" if sdk else "needs the SDK"
    assert r.returncode == 2 and expected in r.stderr, r.stderr
    assert not (tmp_path / "runs").exists()
