"""LIVE Gemini smoke test. Makes a real, possibly paid API call. Excluded by default.

Run only when authorized:

    GEMINI_API_KEY=... LIVE_GEMINI_MODEL=<model id from current docs> \
    LIVE_VIDEO_PATH=/path/to/short_clip.mp4 pytest -m live tests/test_live_gemini.py

It asserts software properties (a response is persisted, the run completes, the output is
parseable or recorded as invalid), never that the model answers correctly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[1]
REQUIRED = ("GEMINI_API_KEY", "LIVE_GEMINI_MODEL", "LIVE_VIDEO_PATH")


@pytest.fixture
def live_env() -> dict[str, str]:
    missing = [k for k in REQUIRED if not os.environ.get(k)]
    if missing:
        pytest.skip(f"live test needs {missing}")
    pytest.importorskip("google.genai")
    return {k: os.environ[k] for k in REQUIRED}


def test_live_whole_report_smoke(tmp_path: Path, live_env: dict[str, str]) -> None:
    from video_report.evaluation.evaluate import evaluate_run
    from video_report.pipeline import create_run, execute_run

    video = Path(live_env["LIVE_VIDEO_PATH"]).resolve()
    bench = tmp_path / "bench"
    bench.mkdir()
    for name in ("manifest.json", "ground_truth.jsonl"):
        (bench / name).write_text((REPO / "benchmarks/mock_mcq_v1" / name).read_text())
    report = json.loads((REPO / "benchmarks/mock_mcq_v1/reports.jsonl").read_text())
    report["input"]["videos"][0].update(uri=str(video), time_range=None)
    (bench / "reports.jsonl").write_text(json.dumps(report) + "\n")

    exp = tmp_path / "exp"
    (exp / "prompts").mkdir(parents=True)
    src = REPO / "experiments/exp003_whole_gemini_smoke"
    for f in ("system.txt", "user.txt"):
        (exp / "prompts" / f).write_text((src / "prompts" / f).read_text())
    cfg = yaml.safe_load((src / "config.yaml").read_text())
    cfg["data"]["benchmark"] = str(bench)
    cfg["provider"]["model"] = live_env["LIVE_GEMINI_MODEL"]
    cfg["execution"].update(concurrency=1, max_attempts=2)
    (exp / "config.yaml").write_text(yaml.safe_dump(cfg))

    run = create_run(exp / "config.yaml", tmp_path / "runs", run_id="live")
    result = execute_run(run, mode="create")
    assert result.state == "completed", run.read_all_calls()
    call = run.read_all_calls()[0]
    payload = run.read_response(call["attempts"][-1]["response_file"])
    assert payload["response"]["text"] is not None
    _, metrics = evaluate_run(run.path, bench / "ground_truth.jsonl")
    assert metrics["fields"]["field_accuracy"]["denominator"] == 3
    assert metrics["execution"]["execution_completion"]["value"] == 1.0
