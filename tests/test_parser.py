"""Response parsing: valid JSON, missing/extra fields, invalid options, duplicates, empties."""

from __future__ import annotations

import pytest

from video_report.benchmarks.schema import FieldSpec, FieldTarget
from video_report.evaluation.parser import parse_response
from video_report.evaluation.scorers import score_field

REQ = ["f1", "f2"]


def test_valid_json() -> None:
    p = parse_response('{"f1": "A", "f2": "C"}', REQ)
    assert p.status == "ok" and p.answers == {"f1": "A", "f2": "C"}
    assert p.missing == () and p.extra_keys == ()


def test_surrounding_whitespace_ok() -> None:
    assert parse_response('  \n{"f1": "A", "f2": "B"}\n ', REQ).status == "ok"


def test_missing_field() -> None:
    p = parse_response('{"f1": "A"}', REQ)
    assert p.status == "ok" and p.missing == ("f2",) and "f2" not in p.answers


def test_extra_fields_recorded_not_fatal() -> None:
    p = parse_response('{"f1": "A", "f2": "B", "f9": "C", "note": "x"}', REQ)
    assert p.status == "ok" and p.extra_keys == ("f9", "note")
    assert p.answers == {"f1": "A", "f2": "B"}


@pytest.mark.parametrize(
    "text, status",
    [
        ("", "empty"),
        ("   \n ", "empty"),
        (None, "empty"),
        ("not json at all", "invalid_json"),
        ('{"f1": "A",}', "invalid_json"),
        ('{"f1": "A"} trailing', "invalid_json"),
        ('["A", "B"]', "not_object"),
        ('"A"', "not_object"),
        ('{"f1": "A", "f1": "B", "f2": "C"}', "duplicate_keys"),
    ],
)
def test_whole_response_failures(text: str | None, status: str) -> None:
    p = parse_response(text, REQ)
    assert p.status == status and p.answers == {}


def test_markdown_fence_rejected_by_default_and_allowed_when_configured() -> None:
    fenced = '```json\n{"f1": "A", "f2": "B"}\n```'
    assert parse_response(fenced, REQ).status == "invalid_json"
    p = parse_response(fenced, REQ, allow_markdown_fence=True)
    assert p.status == "ok" and p.answers == {"f1": "A", "f2": "B"}


# --------------------------------------------------------------------------- field validity

SPEC = FieldSpec(id="f1", question="q?", answer_type="single_choice", choices=["w", "x", "y"])
TARGET = FieldTarget(report_id="r", field_id="f1", target="B")


@pytest.mark.parametrize(
    "value, outcome",
    [
        ("B", "correct"),
        ("A", "incorrect"),
        ("C", "incorrect"),
        ("D", "invalid_answer"),  # letter beyond the 3 choices
        ("Z", "invalid_answer"),
        ("b", "invalid_answer"),  # lowercase is not accepted
        (" B", "invalid_answer"),
        ("B.", "invalid_answer"),
        ("x", "invalid_answer"),  # choice text instead of letter
        (1, "invalid_answer"),
        (None, "invalid_answer"),
        (["B"], "invalid_answer"),
    ],
)
def test_option_validity(value: object, outcome: str) -> None:
    import json

    parsed = parse_response(json.dumps({"f1": value}), ["f1"])
    s = score_field(SPEC, TARGET, "c", parsed)
    assert s.outcome == outcome
    assert s.correct is (outcome == "correct")
    if outcome in ("correct", "incorrect"):
        assert s.predicted_text == SPEC.choices[ord(value) - 65]  # type: ignore[arg-type]


def test_missing_and_invalid_output_and_not_executed() -> None:
    assert score_field(SPEC, TARGET, "c", parse_response("{}", ["f1"])).outcome == "missing_answer"
    assert score_field(SPEC, TARGET, "c", parse_response("", ["f1"])).outcome == "invalid_output"
    s = score_field(SPEC, TARGET, "c", None, "call failed")
    assert s.outcome == "not_executed" and not s.correct and s.detail == "call failed"
