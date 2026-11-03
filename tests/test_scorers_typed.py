"""Scoring rules per answer type (docs/scoring_rules.md), including not_visible and open_text."""

from __future__ import annotations

import json
from typing import Any

import pytest

from video_report.benchmarks.schema import FieldSpec, FieldTarget
from video_report.evaluation.judge import JudgeInput, JudgeVerdict, build_prompt, parse_verdict
from video_report.evaluation.parser import parse_response
from video_report.evaluation.scorers import FieldScore, score_field

TAGS = ["Prep", "Timing", "Technique", "No mistake"]


def spec(answer_type: str, **kw: Any) -> FieldSpec:
    choices = TAGS if answer_type in ("single_choice", "multi_choice") else []
    return FieldSpec(id="f", question="q?", answer_type=answer_type, choices=choices, **kw)


def run(
    sp: FieldSpec, target: Any, value: Any, evidence: dict | None = None, judge=None
) -> FieldScore:
    t = FieldTarget(report_id="r", field_id="f", target=target, evidence=evidence or {})
    return score_field(sp, t, "c", parse_response(json.dumps({"f": value}), ["f"]), judge=judge)


@pytest.mark.parametrize(
    "answer_type, target, value, outcome",
    [
        ("bool", True, True, "correct"),
        ("bool", True, False, "incorrect"),
        ("bool", True, "true", "invalid_answer"),
        ("bool", True, 1, "invalid_answer"),
        ("int", 3, 3, "correct"),
        ("int", 3, 4, "incorrect"),
        ("int", 3, 3.0, "invalid_answer"),
        ("int", 3, True, "invalid_answer"),
        ("single_choice", "B", "B", "correct"),
        ("single_choice", ["A", "C"], "C", "correct"),  # any one of several
        ("single_choice", ["A", "C"], "B", "incorrect"),
        ("single_choice", "B", ["B"], "invalid_answer"),
        ("multi_choice", ["A", "C"], ["C", "A"], "correct"),
        ("multi_choice", ["A", "C"], ["A"], "incorrect"),
        ("multi_choice", ["A"], ["A", "A"], "invalid_answer"),
        ("multi_choice", ["A"], ["Z"], "invalid_answer"),
        ("multi_choice", [], [], "correct"),
    ],
)
def test_simple_types(answer_type: str, target: Any, value: Any, outcome: str) -> None:
    s = run(spec(answer_type), target, value)
    assert s.outcome == outcome and s.correct is (outcome == "correct")


def test_multi_choice_counts_and_exclusive_option() -> None:
    s = run(spec("multi_choice"), ["A", "C"], ["C", "D"])
    assert s.extra | {} == {"answer_type": "multi_choice", "hits": 1, "misses": 1, "extras": 1}
    sp = spec("multi_choice", exclusive_choice="No mistake")
    assert run(sp, ["D"], ["D"]).outcome == "correct"
    assert run(sp, ["D"], []).outcome == "invalid_answer"
    assert run(sp, ["D"], ["D", "A"]).outcome == "invalid_answer"
    assert run(sp, ["A"], ["A", "B"]).outcome == "incorrect"


@pytest.mark.parametrize(
    "target, value, ok",
    [
        (4, 2.0, True),
        (4, 6.0, True),
        (4, 6.5, False),
        (40, 50, True),
        (40, 51, False),
        (0, 2, True),
    ],
)
def test_seconds_tolerance(target: float, value: float, ok: bool) -> None:
    s = run(spec("seconds"), target, value)
    assert s.correct is ok and s.extra["abs_error"] == abs(value - target)


def test_seconds_validity_and_timestamp_tolerance() -> None:
    assert run(spec("seconds"), 10, -1).outcome == "invalid_answer"
    assert run(spec("seconds"), 10, "10").outcome == "invalid_answer"
    # a timestamp at 600 s with a 20 s step: tolerance max(2, 5) = 5, not 150
    ts = spec("seconds", timestamp=True)
    assert (
        run(ts, 600, 604, {"length": 20}).correct and not run(ts, 600, 606, {"length": 20}).correct
    )


def test_not_visible_rules() -> None:
    nv = spec("bool", allow_not_visible=True)
    hall = run(nv, "not_visible", True)  # an answer for a step that was not done
    assert hall.outcome == "incorrect" and hall.extra["hallucination"]
    assert run(nv, "not_visible", "not_visible").correct
    missed = run(nv, True, "not_visible")  # not_visible on a filmed step
    assert missed.outcome == "incorrect" and missed.extra["false_not_done"]
    assert run(spec("bool"), True, "not_visible").outcome == "invalid_answer"  # not allowed
    for t in ("int", "seconds", "single_choice", "multi_choice"):
        sp = spec(t, allow_not_visible=True)
        assert run(sp, "not_visible", "not_visible").correct
        assert run(sp, "not_visible", 1 if t in ("int", "seconds") else "A").extra.get(
            "hallucination"
        ) in (True, None)


def test_open_text_with_judge() -> None:
    calls: list[JudgeInput] = []

    def judge(inp: JudgeInput) -> JudgeVerdict:
        calls.append(inp)
        return JudgeVerdict(True, [0], 1)

    sp = spec("open_text")
    ref = ["Used 1 minute instead of 1.5"]
    s = run(sp, ref, "It was microwaved for one minute", judge=judge)
    assert s.correct and s.extra["matched"] == [0] and s.extra["extra_claims"] == 1
    assert run(sp, [], "None.", judge=judge).correct  # "none" for a clean step, no judge call
    assert not run(sp, ref, "none", judge=judge).correct
    assert len(calls) == 1
    assert run(sp, ref, "", judge=judge).outcome == "invalid_answer"


def test_judge_prompt_and_reply_parsing() -> None:
    p = build_prompt(JudgeInput("What differed?", ["too much salt"], "salty", ["extra"]))
    assert "0. too much salt" in p and "Answer to grade: salty" in p
    v = parse_verdict('{"correct": true, "matched": [0, 0], "extra_claims": 2}', 1)
    assert v == JudgeVerdict(True, [0], 2)
    # no reference: judges write ["none"] for matched
    none = '{"correct": true, "matched": ["none"], "extra_claims": 0}'
    assert parse_verdict(none, 0).matched == []
    for bad in ('{"correct": "yes", "matched": [], "extra_claims": 0}',
                '{"correct": true, "matched": [3], "extra_claims": 0}',
                '{"correct": true, "matched": [], "extra_claims": -1}'):  # fmt: skip
        with pytest.raises(ValueError):
            parse_verdict(bad, 1)


class FakeProvider:
    def __init__(self, replies: list[str]) -> None:
        self.replies, self.requests = replies, []

    def generate(self, request, *, timeout_s: float):  # type: ignore[no-untyped-def]
        from video_report.providers.base import ProviderResponse

        self.requests.append(request)
        return ProviderResponse(text=self.replies.pop(0), usage=None)


def test_provider_judge_retries_bad_json_and_caches(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from video_report.evaluation.judge import ProviderJudge

    inp = JudgeInput("q?", ["too salty"], "salty", [])
    good = '{"correct": true, "matched": [0], "extra_claims": 0}'
    provider = FakeProvider(["not json", good])
    cache = tmp_path / "judge_cache.jsonl"
    judge = ProviderJudge(provider, "m", cache_path=cache, sleep=lambda s: None)
    assert judge(inp) == JudgeVerdict(True, [0], 0)
    assert judge(inp).correct and len(provider.requests) == 2  # the second answer is cached
    assert provider.requests[0].videos == () and "salty" in provider.requests[0].user
    again = ProviderJudge(FakeProvider([]), "m", cache_path=cache)  # new process, same cache file
    assert again(inp).correct and again.stats["cache_hits"] == 1
    failing = ProviderJudge(FakeProvider(["x", "x", "x"]), "m", sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="judge failed after 3 attempts"):
        failing(inp)


def test_review_fixes() -> None:
    from video_report.datasets.base import DataValidationError, validate_targets
    from video_report.evaluation.aggregate import aggregate

    # every score carries answer_type / not_done_item, also with no parsed answer
    nv = spec("bool", allow_not_visible=True)
    t = FieldTarget(report_id="r", field_id="f", target="not_visible")
    gone = score_field(nv, t, "c", parse_response("{}", ["f"]))
    assert gone.outcome == "missing_answer" and gone.extra == {
        "answer_type": "bool",
        "not_done_item": True,
    }
    failed = score_field(spec("int"), FieldTarget(report_id="r", field_id="f", target=1), "c", None)
    m = aggregate([gone, failed], [], None, writer_active=False)["fields"]
    assert set(m["by_answer_type"]) == {"bool", "int"}
    assert m["not_done"]["items"] == 1 and m["not_done"]["hallucination"]["denominator"] == 1
    # huge integers are invalid, not a crash
    assert run(spec("seconds"), 10, 10**400).outcome == "invalid_answer"
    # several accepted letters keep all their texts
    assert run(spec("single_choice"), ["A", "C"], "C").target_text == ["Prep", "Technique"]
    # a timestamp target without the step length is rejected at validation
    from video_report.benchmarks.schema import Report

    rep = Report.model_validate(
        {"id": "r", "input": {"videos": [{"id": "v", "uri": "v.mp4"}]},
         "fields": [{"id": "f", "question": "q?", "answer_type": "seconds", "timestamp": True}],
         "source": {"dataset": "x"}}
    )  # fmt: skip
    with pytest.raises(DataValidationError, match="evidence.length"):
        validate_targets([rep], [FieldTarget(report_id="r", field_id="f", target=5)], {"r": ["f"]})
    # judge replies that are not objects are a ValueError (retried), not a TypeError
    for bad in ('[{"correct": true}]', '"yes"', '{"correct": true}'):
        with pytest.raises(ValueError):
            parse_verdict(bad, 1)
