"""Non-single_choice answer types: field rules, target validation, answer-format rendering."""

from __future__ import annotations

from typing import Any

import pytest

from conftest import make_report
from video_report.benchmarks.schema import MAX_CHOICES, FieldTarget, choice_letter, letter_index
from video_report.datasets.base import DataValidationError, parse_report, validate_targets
from video_report.prompting import render_answer_format, render_questions

TAGS = ["Technique", "Order", "Missing Step"]


def _report(allow_not_visible: bool = False) -> dict:
    r = make_report("r1", ["f1"])
    r["fields"] = [
        {"id": "ok", "question": "Done correctly?", "answer_type": "bool"},
        {"id": "n", "question": "How many?", "answer_type": "int"},
        {"id": "dur", "question": "How long?", "answer_type": "seconds"},
        {"id": "tags", "question": "Mistakes?", "answer_type": "multi_choice", "choices": TAGS},
        {"id": "any", "question": "Which?", "answer_type": "single_choice", "choices": TAGS},
        {"id": "txt", "question": "What differed?", "answer_type": "open_text"},
    ]
    for f in r["fields"]:
        f["allow_not_visible"] = allow_not_visible
    return r


def _check(targets: dict[str, Any], allow_not_visible: bool = False) -> None:
    report = parse_report(_report(allow_not_visible))
    rows = [FieldTarget(report_id="r1", field_id=k, target=v) for k, v in targets.items()]
    validate_targets([report], rows, {"r1": list(targets)})


GOOD = {
    "ok": False,
    "n": 3,
    "dur": 42.5,
    "tags": ["B", "A"],
    "any": ["A", "C"],
    "txt": ["too much salt"],
}


def test_valid_targets_per_type() -> None:
    _check(GOOD)
    _check({**GOOD, "dur": 0, "tags": [], "any": "B", "txt": []})


@pytest.mark.parametrize(
    "field, bad",
    [
        ("ok", "true"),
        ("ok", 1),
        ("n", 2.5),
        ("n", True),
        ("dur", -1),
        ("dur", "12"),
        ("tags", "A"),
        ("tags", ["A", "A"]),
        ("tags", ["D"]),
        ("any", []),
        ("any", ["A", "A"]),
        ("any", ["D"]),
        ("txt", "too much salt"),
        ("txt", [""]),
        ("ok", "not_visible"),
    ],
)
def test_invalid_targets_rejected(field: str, bad: Any) -> None:
    with pytest.raises(DataValidationError, match=f"target for \\(r1, {field}\\)"):
        _check({**GOOD, field: bad})


def test_not_visible_only_when_allowed() -> None:
    _check({k: "not_visible" for k in GOOD}, allow_not_visible=True)


def test_exclusive_choice_and_timestamp() -> None:
    r = _report()
    r["fields"][3]["exclusive_choice"] = "Missing Step"
    r["fields"][2]["timestamp"] = True
    report = parse_report(r)
    ok = [FieldTarget(report_id="r1", field_id="tags", target=["C"])]
    validate_targets([report], ok, {"r1": ["tags"]})
    for bad in ([], ["C", "A"]):
        rows = [FieldTarget(report_id="r1", field_id="tags", target=bad)]
        with pytest.raises(DataValidationError, match="answered alone"):
            validate_targets([report], rows, {"r1": ["tags"]})
    assert 'C ("Missing Step") only on its own>' in render_answer_format(report.fields)
    r["fields"][0]["timestamp"] = True  # bool cannot be a timestamp
    with pytest.raises(DataValidationError, match="timestamp is only valid"):
        parse_report(r)
    r = _report()
    r["fields"][3]["exclusive_choice"] = "nope"
    with pytest.raises(DataValidationError, match="exclusive_choice"):
        parse_report(r)


def test_choices_rules() -> None:
    r = _report()
    r["fields"][0]["choices"] = ["yes", "no"]
    with pytest.raises(DataValidationError, match="bool takes no choices"):
        parse_report(r)
    r = _report()
    r["fields"][3]["choices"] = ["only"]
    with pytest.raises(DataValidationError, match="multi_choice needs 2..702"):
        parse_report(r)


def test_choice_labels_round_trip() -> None:
    labels = [choice_letter(i) for i in range(MAX_CHOICES)]
    assert labels[:3] == ["A", "B", "C"] and labels[25:28] == ["Z", "AA", "AB"]
    assert labels[-1] == "ZZ" and len(set(labels)) == MAX_CHOICES
    assert [letter_index(x) for x in labels] == list(range(MAX_CHOICES))
    assert [letter_index(x) for x in ["", "a", "Aa", "AAA", "1"]] == [None] * 5


def test_rendering_per_type() -> None:
    fields = parse_report(_report(allow_not_visible=True)).fields
    # bare JSON types are not quoted (a quoted placeholder made models answer "9" or "F");
    # letters and text are
    assert render_answer_format(fields) == "\n".join(
        [
            "{",
            '  "ok": <true or false or "not_visible">,',
            '  "n": <integer or "not_visible">,',
            '  "dur": <number of seconds or "not_visible">,',
            '  "tags": <JSON list of letters from A/B/C, e.g. ["A", "C"], [] if none'
            ' or "not_visible">,',
            '  "any": "<one letter: A/B/C or \\"not_visible\\">",',
            '  "txt": "<short text answer or \\"not_visible\\">"',
            "}",
        ]
    )
    text = render_questions(fields)
    assert "Q1. [field_id: ok]\nQuestion: Done correctly?\n\nQ2." in text
    assert "Options:\nA. Technique\nB. Order\nC. Missing Step" in text
