"""Data contract validation: IDs, choices, targets, time ranges, answer types, leakage keys."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from conftest import make_report, write_jsonl
from video_report.benchmarks.schema import FieldTarget
from video_report.datasets.base import (
    DataValidationError,
    choose_video_root,
    load_benchmark,
    load_ground_truth,
    load_reports,
    parse_report,
    resolve_video,
    validate_targets,
)


def _report(**changes: object) -> dict:
    r = make_report("r1", ["f1", "f2"])
    r.update(changes)
    return r


def test_committed_fixture_is_valid() -> None:
    bench = load_benchmark(Path(__file__).resolve().parents[1] / "benchmarks" / "mock_mcq_v1")
    reports = load_reports(bench.reports_path)
    assert [r.id for r in reports] == ["synthetic-0001"]
    assert [f.id for f in reports[0].fields] == [
        "primary_appliance",
        "first_item_in_cup",
        "task_outcome",
    ]
    assert bench.ground_truth_path is not None
    sel = {r.id: [f.id for f in r.fields] for r in reports}
    assert len(validate_targets(reports, load_ground_truth(bench.ground_truth_path), sel)) == 3


def test_duplicate_report_ids_rejected(tmp_path: Path) -> None:
    p = write_jsonl(tmp_path / "r.jsonl", [_report(), _report()])
    with pytest.raises(DataValidationError, match="duplicate report id"):
        load_reports(p)


def test_duplicate_field_ids_rejected() -> None:
    r = _report()
    r["fields"].append(copy.deepcopy(r["fields"][0]))
    with pytest.raises(DataValidationError, match="duplicate field ids"):
        parse_report(r)


def test_duplicate_video_ids_rejected() -> None:
    r = _report()
    r["input"]["videos"].append(copy.deepcopy(r["input"]["videos"][0]))
    with pytest.raises(DataValidationError, match="duplicate video ids"):
        parse_report(r)


def test_duplicate_json_keys_in_reports_file_rejected(tmp_path: Path) -> None:
    p = tmp_path / "r.jsonl"
    p.write_text('{"id": "a", "id": "b"}\n')
    with pytest.raises(DataValidationError, match="duplicate JSON key"):
        load_reports(p)


@pytest.mark.parametrize("bad_id", ["", "has space", "-leading", "a/b", "x" * 129, "a::b"])
def test_invalid_ids_rejected(bad_id: str) -> None:
    with pytest.raises(DataValidationError, match="invalid id"):
        parse_report(_report(id=bad_id))


@pytest.mark.parametrize(
    "choices, msg",
    [
        (["only one"], "2..26 choices"),
        ([f"c{i}" for i in range(27)], "2..26 choices"),
        (["same", "same"], "duplicate choices"),
        (["same", " same "], "duplicate choices"),
        (["ok", "   "], "non-empty"),
    ],
)
def test_invalid_choices_rejected(choices: list[str], msg: str) -> None:
    r = _report()
    r["fields"][0]["choices"] = choices
    with pytest.raises(DataValidationError, match=msg):
        parse_report(r)


@pytest.mark.parametrize("answer_type", ["multiple_choice", "free_text", "number", "boolean", ""])
def test_unsupported_answer_types_rejected(answer_type: str) -> None:
    r = _report()
    r["fields"][1]["answer_type"] = answer_type
    with pytest.raises(DataValidationError, match="unsupported answer_type"):
        parse_report(r)


@pytest.mark.parametrize(
    "time_range, msg",
    [
        ({"start": 5, "end": 5, "unit": "seconds", "reference": "video_start"}, "must be >"),
        ({"start": 9, "end": 3, "unit": "seconds", "reference": "video_start"}, "must be >"),
        ({"start": -1, "end": 3, "unit": "seconds", "reference": "video_start"}, "greater than"),
        ({"start": 0, "end": 3, "unit": "frames", "reference": "video_start"}, "unit"),
        ({"start": 0, "end": 3, "unit": "seconds", "reference": "wall_clock"}, "reference"),
        ({"start": 0, "end": 3}, "unit"),
        (
            {"start": 0, "end": float("inf"), "unit": "seconds", "reference": "video_start"},
            "finite",
        ),
    ],
)
def test_invalid_time_ranges_rejected(time_range: dict, msg: str) -> None:
    r = _report()
    r["input"]["videos"][0]["time_range"] = time_range
    with pytest.raises(DataValidationError, match=msg):
        parse_report(r)


def test_valid_time_range_accepted() -> None:
    r = _report()
    r["input"]["videos"][0]["time_range"] = {
        "start": 0,
        "end": 2.5,
        "unit": "seconds",
        "reference": "video_start",
    }
    assert parse_report(r).input.videos[0].time_range is not None


@pytest.mark.parametrize("leak_key", ["target", "answer", "evidence"])
def test_target_like_keys_in_reports_rejected(leak_key: str) -> None:
    r = _report()
    r["fields"][0][leak_key] = "A"
    with pytest.raises(DataValidationError, match="Extra inputs are not permitted"):
        parse_report(r)
    r2 = _report(**{leak_key: "A"})
    with pytest.raises(DataValidationError, match="Extra inputs are not permitted"):
        parse_report(r2)


# --------------------------------------------------------------------------- targets


def _targets(rows: list[dict]) -> list[FieldTarget]:
    return [FieldTarget.model_validate(r) for r in rows]


def _validate(rows: list[dict]) -> dict:
    report = parse_report(_report())
    return validate_targets([report], _targets(rows), {"r1": ["f1", "f2"]})


def test_valid_targets() -> None:
    idx = _validate(
        [
            {"report_id": "r1", "field_id": "f1", "target": "A", "target_text": "f1 option 0"},
            {"report_id": "r1", "field_id": "f2", "target": "D"},
        ]
    )
    assert idx[("r1", "f2")].target == "D"


@pytest.mark.parametrize("bad", ["E", "a", "AB", "", "1"])
def test_invalid_target_letters_rejected(bad: str) -> None:
    rows = [
        {"report_id": "r1", "field_id": "f1", "target": bad},
        {"report_id": "r1", "field_id": "f2", "target": "A"},
    ]
    with pytest.raises(DataValidationError, match="expected one of"):
        _validate(rows)


def test_target_text_mismatch_rejected() -> None:
    rows = [
        {"report_id": "r1", "field_id": "f1", "target": "B", "target_text": "f1 option 0"},
        {"report_id": "r1", "field_id": "f2", "target": "A"},
    ]
    with pytest.raises(DataValidationError, match="target_text mismatch"):
        _validate(rows)


def test_missing_target_for_selected_field_rejected() -> None:
    with pytest.raises(DataValidationError, match="missing targets"):
        _validate([{"report_id": "r1", "field_id": "f1", "target": "A"}])


def test_target_for_unknown_field_rejected() -> None:
    rows = [
        {"report_id": "r1", "field_id": "f1", "target": "A"},
        {"report_id": "r1", "field_id": "f2", "target": "A"},
        {"report_id": "r1", "field_id": "nope", "target": "A"},
    ]
    with pytest.raises(DataValidationError, match="unknown field 'nope'"):
        _validate(rows)


def test_duplicate_ground_truth_rows_rejected(tmp_path: Path) -> None:
    row = {"report_id": "r1", "field_id": "f1", "target": "A"}
    p = write_jsonl(tmp_path / "gt.jsonl", [row, row])
    with pytest.raises(DataValidationError, match="duplicate target"):
        load_ground_truth(p)


# --------------------------------------------------------------------------- video paths


def test_video_root_precedence(tmp_path: Path, mock_bench: Path) -> None:
    bench = load_benchmark(mock_bench)
    cfg_root = tmp_path / "cfg_videos"
    assert choose_video_root("/cli", cfg_root, bench, {"VIDEO_REPORT_VIDEO_ROOT": "/env"}) == (
        Path("/cli").resolve(),
        "cli",
    )
    assert choose_video_root(None, cfg_root, bench, {"VIDEO_REPORT_VIDEO_ROOT": "/env"})[1] == "env"
    assert choose_video_root(None, cfg_root, bench, {})[1] == "config"
    assert choose_video_root(None, None, bench, {}) == (bench.root, "benchmark_dir")


def test_resolve_video_kinds(tmp_path: Path) -> None:
    rel = resolve_video("v", "clips/a.mp4", None, tmp_path)
    assert rel.is_local and rel.location == str((tmp_path / "clips/a.mp4").resolve())
    assert rel.mime_type == "video/mp4"
    remote = resolve_video("v", "gs://bucket/a.mp4", None, tmp_path)
    assert not remote.is_local and remote.location == "gs://bucket/a.mp4"
    with pytest.raises(DataValidationError, match="no video root"):
        resolve_video("v", "a.mp4", None, None)
