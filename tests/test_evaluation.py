"""Offline evaluation of runs: end-to-end mock flow, partial / failed runs, immutability."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from conftest import ScriptedProvider
from video_report.datasets.base import DataValidationError
from video_report.evaluation.evaluate import evaluate_run
from video_report.io_utils import sha256_file
from video_report.pipeline import create_run, execute_run
from video_report.providers.base import Usage

REPO = Path(__file__).resolve().parents[1]
GT = REPO / "benchmarks/mock_mcq_v1/ground_truth.jsonl"


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and "evaluations" not in p.parts and p.name != "run.lock"
    }


@pytest.mark.parametrize(
    "exp, expected",
    [
        # Hand-checked against benchmarks/mock_mcq_v1: targets A, A, C.
        # whole: mock answers A, C, C -> correct, incorrect, correct
        ("exp001_whole_mcq", {"correct": 2, "incorrect": 1, "invalid_output": 0}),
        # per-field: same answers but task_outcome's raw text is not JSON -> invalid_output
        ("exp002_per_field_mcq", {"correct": 1, "incorrect": 1, "invalid_output": 1}),
    ],
)
def test_mock_inference_then_offline_scoring(
    tmp_path: Path, exp: str, expected: dict[str, int]
) -> None:
    run = create_run(REPO / f"experiments/{exp}/config.yaml", tmp_path, run_id="r", environ={})
    assert execute_run(run, mode="create").state == "completed"
    before = _tree_hashes(run.path)

    out1, m = evaluate_run(run.path, GT)
    for k, v in expected.items():
        assert m["fields"]["outcomes"][k] == v
    acc = m["fields"]["field_accuracy"]
    assert acc["denominator"] == 3 and acc["numerator"] == expected["correct"]
    assert m["reports"]["all_fields_correct_rate"]["value"] == 0.0
    assert m["reports"]["report_completeness_rate"]["value"] == (
        1.0 if expected["invalid_output"] == 0 else 0.0
    )
    assert m["provisional"] is False
    assert m["usage"]["cost_status"] == "unknown" and m["usage"]["cost"] is None  # mock usage

    # Second evaluation goes to a new directory; inference evidence is untouched.
    out2, m2 = evaluate_run(run.path, GT, allow_markdown_fence=True)
    assert out1 != out2 and out1.is_dir() and out2.is_dir()
    assert _tree_hashes(run.path) == before
    ev = json.loads((out2 / "evaluation.json").read_text())
    assert ev["scoring"]["parser_version"] == "json_letters_v1"
    assert ev["scoring"]["allow_markdown_fence"] is True
    assert ev["ground_truth"]["file_sha256"] == sha256_file(GT)
    assert ev["ground_truth"]["num_selected_targets"] == 3
    rows = [json.loads(x) for x in (out1 / "field_scores.jsonl").read_text().splitlines()]
    assert len(rows) == 3


def _run(
    tmp_path: Path,
    make_bench: Callable[..., Path],
    make_experiment: Callable[..., Path],
    provider: ScriptedProvider,
    **cfg: Any,
) -> Path:
    bench = make_bench(
        {"r1": ["f1", "f2"], "r2": ["f1"]},
        targets={("r1", "f1"): "A", ("r1", "f2"): "B", ("r2", "f1"): "A"},
    )
    run = create_run(
        make_experiment(bench, method="per_field", **cfg), tmp_path / "runs", run_id="r", environ={}
    )
    execute_run(run, mode="create", provider_factory=lambda c, r: provider, sleep=lambda s: None)
    return bench


def test_partial_run_is_provisional_with_denominator(
    tmp_path: Path, make_bench: Callable[..., Path], make_experiment: Callable[..., Path]
) -> None:
    provider = ScriptedProvider(
        script={"r1::field=f2": ["permanent"]}, usage=Usage(input_tokens=1000, output_tokens=10)
    )
    bench = _run(
        tmp_path,
        make_bench,
        make_experiment,
        provider,
        pricing={"input_per_million": 2.0, "output_per_million": 10.0, "source": "test"},
    )
    _, m = evaluate_run(tmp_path / "runs/r", bench / "ground_truth.jsonl")
    # 3 fields: r1/f1 A==A correct, r1/f2 not executed, r2/f1 A==A correct -> 2/3
    assert m["fields"]["field_accuracy"] == {"value": 2 / 3, "numerator": 2, "denominator": 3}
    assert m["fields"]["outcomes"]["not_executed"] == 1
    assert m["provisional"] is True and "1 of 3 calls" in m["provisional_reasons"][0]
    assert m["execution"]["execution_completion"]["numerator"] == 2
    assert m["execution"]["failed_calls_by_reason"] == {"permanent_error": 1}
    # cost: 2 successful attempts x (1000*2 + 10*10)/1e6 = 2 x 0.0021 = 0.0042
    assert m["usage"]["cost"] == pytest.approx(0.0042)
    # only r2 is a completed report: 0.0021
    assert m["usage"]["cost_per_completed_report"]["mean"] == pytest.approx(0.0021)
    assert m["usage"]["completed_reports"] == 1


def test_all_failed_run_accounting(
    tmp_path: Path, make_bench: Callable[..., Path], make_experiment: Callable[..., Path]
) -> None:
    provider = ScriptedProvider(
        script={c: ["timeout"] * 9 for c in ("r1::field=f1", "r1::field=f2", "r2::field=f1")}
    )
    bench = _run(
        tmp_path,
        make_bench,
        make_experiment,
        provider,
        execution={"max_attempts": 2, "backoff_initial_s": 0.0},
        pricing={"input_per_million": 2.0, "output_per_million": 10.0, "source": "test"},
    )
    _, m = evaluate_run(tmp_path / "runs/r", bench / "ground_truth.jsonl")
    assert m["fields"]["field_accuracy"]["value"] == 0.0
    assert m["fields"]["outcomes"]["not_executed"] == 3
    assert m["execution"]["attempts_total"] == 6
    assert m["execution"]["attempt_outcomes"] == {"timeout": 6}
    assert m["usage"]["cost"] is None and m["usage"]["cost_status"] == "unknown"
    assert m["warnings"] and m["usage"]["note"]


def test_evaluation_requires_valid_targets(
    tmp_path: Path, make_bench: Callable[..., Path], make_experiment: Callable[..., Path]
) -> None:
    bench = _run(tmp_path, make_bench, make_experiment, ScriptedProvider())
    gt = bench / "ground_truth.jsonl"
    rows = gt.read_text().splitlines()
    gt.write_text("\n".join(rows[:-1]) + "\n")  # drop the r2/f1 target
    with pytest.raises(DataValidationError, match="missing targets"):
        evaluate_run(tmp_path / "runs/r", gt)
    assert not (tmp_path / "runs/r/evaluations").exists()


def test_field_subset_run_ignores_targets_of_unselected_fields(tmp_path: Path) -> None:
    import yaml

    exp = tmp_path / "exp"
    shutil.copytree(REPO / "experiments/exp001_whole_mcq", exp)
    cfg = yaml.safe_load((exp / "config.yaml").read_text())
    cfg["data"].update(
        benchmark=str(REPO / "benchmarks/mock_mcq_v1"), field_ids=["primary_appliance"]
    )
    (exp / "config.yaml").write_text(yaml.safe_dump(cfg))
    run = create_run(exp / "config.yaml", tmp_path / "runs", run_id="sub", environ={})
    execute_run(run, mode="create")
    _, m = evaluate_run(run.path, GT)  # GT also has targets for the 2 unselected fields
    assert m["fields"]["field_accuracy"] == {"value": 1.0, "numerator": 1, "denominator": 1}
