import json
import os
import subprocess
import sys

import pytest

from video_report.benchmarks.baseline import validate_pilot_budget, verify_derived_ground_truth


def test_valid_gt_and_consistent_but_changed_answer(make_bench, tmp_path):
    source = make_bench({"r": ["f"]})
    derived = tmp_path / "derived.jsonl"
    gt = source / "ground_truth.jsonl"
    derived.write_bytes(gt.read_bytes())
    digest = verify_derived_ground_truth(gt, derived, source / "reports.jsonl")
    assert len(digest) == 64
    row = json.loads(derived.read_text())
    report = json.loads((source / "reports.jsonl").read_text())
    row["target"] = "B"
    row["target_text"] = report["fields"][0]["choices"][1]
    derived.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="source"):
        verify_derived_ground_truth(gt, derived, source / "reports.jsonl")


def test_missing_gt_is_rejected(make_bench, tmp_path):
    source = make_bench({"r": ["f"]})
    derived = tmp_path / "empty.jsonl"
    derived.write_text("")
    with pytest.raises(ValueError):
        verify_derived_ground_truth(
            source / "ground_truth.jsonl", derived, source / "reports.jsonl"
        )


@pytest.mark.parametrize(
    "calls,cost,limit",
    [(123, 0.5, 1), (125, 0.5, 1), (124, 2, 1), (124, float("nan"), 1), (124, 0.5, -1)],
)
def test_invalid_scope_or_budget_is_rejected(calls, cost, limit):
    with pytest.raises(ValueError):
        validate_pilot_budget(calls, cost, limit)


def test_budget_allows_equal_limit():
    validate_pilot_budget(124, 1, 1)


def test_budget_cannot_be_disabled_by_python_optimization():
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "from video_report.benchmarks.baseline import validate_pilot_budget; "
            "validate_pilot_budget(124, 2, 1)",
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "budget" in result.stderr
