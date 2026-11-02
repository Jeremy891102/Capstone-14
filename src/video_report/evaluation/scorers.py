"""Score one field: compare a parsed answer with its target. Pure function; no I/O.

The rules per answer type are in docs/scoring_rules.md. Outcomes (every selected field gets
exactly one; all count in the accuracy denominator):

* ``correct``          valid answer that the type's rule accepts
* ``incorrect``        valid answer that the rule rejects
* ``invalid_answer``   key present but the value is not a valid answer for this field
* ``missing_answer``   response parsed but this field's key is absent
* ``invalid_output``   whole response unusable (empty, invalid JSON, duplicate keys, not object)
* ``not_executed``     no successful provider call for this field (failed / pending / interrupted)

Secondary measurements go in ``FieldScore.extra`` and never change the outcome.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from video_report.benchmarks.schema import NOT_VISIBLE, FieldSpec, FieldTarget, letter_index
from video_report.evaluation.judge import Judge, JudgeInput
from video_report.evaluation.parser import ParsedResponse

SCORER_VERSION = "typed_v2"

Outcome = Literal[
    "correct", "incorrect", "invalid_answer", "missing_answer", "invalid_output", "not_executed"
]
VALID_ANSWER_OUTCOMES = frozenset({"correct", "incorrect"})


@dataclass(frozen=True)
class ScoringConfig:
    """Tolerance for seconds fields; recorded in every evaluation."""

    seconds_floor: float = 2.0
    seconds_ratio: float = 0.25

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FieldScore:
    report_id: str
    field_id: str
    call_id: str
    outcome: Outcome
    correct: bool
    predicted: Any  # the valid answer, if any
    predicted_text: Any
    target: Any
    target_text: Any
    detail: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Verdict:
    outcome: Outcome
    predicted: Any = None
    predicted_text: Any = None
    detail: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v: float) -> bool:
    try:
        return math.isfinite(v)
    except OverflowError:  # an int too large for a float
        return False


def _letters(spec: FieldSpec, raw: Any) -> list[str] | None:
    """The answer as a list of valid, distinct letters, or None. A bare string is one letter."""
    items = raw if isinstance(raw, list) else [raw]
    if len(set(map(str, items))) != len(items):
        return None
    for x in items:
        if not isinstance(x, str) or (i := letter_index(x)) is None or i >= len(spec.choices):
            return None
    return list(items)


def _texts(spec: FieldSpec, letters: list[str]) -> list[str]:
    return [spec.choices[letter_index(x)] for x in letters]  # type: ignore[index]


def ok(correct: bool) -> Outcome:
    return "correct" if correct else "incorrect"


def _invalid(why: str) -> _Verdict:
    return _Verdict("invalid_answer", detail=why)


def _target_set(target: Any) -> set[str]:
    return {target} if isinstance(target, str) else set(target)


def _judge_open_text(
    spec: FieldSpec, target: FieldTarget, raw: str, judge: Judge | None
) -> _Verdict:
    reference = list(target.target)  # type: ignore[arg-type]
    said_none = raw.strip().strip(".").lower() == "none"
    if said_none:  # no judge call needed
        correct = not reference
        return _Verdict("correct" if correct else "incorrect", raw, extra={"matched": []})
    assert judge is not None, "open_text fields need a judge"
    mod = target.evidence.get("modified_description")  # step-level note from the annotators
    notes = [mod] if isinstance(mod, str) and mod.strip() else []
    verdict = judge(JudgeInput(spec.question, reference, raw, notes))
    return _Verdict(
        "correct" if verdict.correct else "incorrect",
        raw,
        extra={"matched": verdict.matched, "extra_claims": verdict.extra_claims},
    )


def _check(
    spec: FieldSpec, target: FieldTarget, raw: Any, cfg: ScoringConfig, judge: Judge | None
) -> _Verdict:
    not_done_item = target.target == NOT_VISIBLE
    extra: dict[str, Any] = {}
    if isinstance(raw, str) and raw == NOT_VISIBLE:
        if not spec.allow_not_visible:
            return _invalid("field does not allow not_visible")
        if not not_done_item:
            extra["false_not_done"] = True
        return _Verdict("correct" if not_done_item else "incorrect", NOT_VISIBLE, extra=extra)

    v = _check_type(spec, target, raw, cfg, judge, not_done_item)
    if v.outcome in VALID_ANSWER_OUTCOMES and not_done_item:
        v.outcome, extra["hallucination"] = "incorrect", True  # anything but not_visible
    v.extra = {**extra, **v.extra}
    return v


def _check_type(
    spec: FieldSpec,
    target: FieldTarget,
    raw: Any,
    cfg: ScoringConfig,
    judge: Judge | None,
    not_done_item: bool,
) -> _Verdict:
    t, tgt = spec.answer_type, target.target
    if t == "bool":
        if not isinstance(raw, bool):
            return _invalid(f"value {raw!r} is not true or false")
        return _Verdict(ok(raw == tgt), raw)
    if t == "int":
        if not isinstance(raw, int) or isinstance(raw, bool):
            return _invalid(f"value {raw!r} is not an integer")
        int_extra: dict[str, Any] = {} if not_done_item else {"abs_error": abs(raw - tgt)}  # type: ignore[operator]
        return _Verdict(ok(raw == tgt), raw, extra=int_extra)
    if t == "seconds":
        if not _is_number(raw) or not _finite(raw) or raw < 0:
            return _invalid(f"value {raw!r} is not a finite number of seconds >= 0")
        if not_done_item:
            return _Verdict("incorrect", raw)
        goal = float(tgt)  # type: ignore[arg-type]
        scale = float(target.evidence["length"]) if spec.timestamp else goal  # validated
        tol = max(cfg.seconds_floor, cfg.seconds_ratio * scale)
        err = abs(raw - goal)
        extra: dict[str, Any] = {
            "abs_error": err,
            "rel_error": err / goal if goal else None,
            "tolerance": tol,
        }
        return _Verdict(ok(err <= tol), raw, extra=extra)
    if t == "open_text":
        if not isinstance(raw, str) or not raw.strip():
            return _invalid(f"value {raw!r} is not a non-empty string")
        if not_done_item:
            return _Verdict("incorrect", raw)
        return _judge_open_text(spec, target, raw, judge)

    letters = _letters(spec, raw)
    if t == "single_choice":
        if not isinstance(raw, str) or not letters:
            return _invalid(f"value {raw!r} is not one of {spec.letters()}")
        accepted = set() if not_done_item else _target_set(tgt)
        return _Verdict(ok(raw in accepted), raw, _texts(spec, letters)[0])
    assert t == "multi_choice", t
    if letters is None or not isinstance(raw, list):
        return _invalid(f"value {raw!r} is not a list of distinct letters from {spec.letters()}")
    ex = spec.exclusive_letter()
    if ex is not None and (not letters or (ex in letters and len(letters) > 1)):
        return _invalid(f"{ex} ({spec.exclusive_choice}) is answered alone; [] is not allowed")
    if not_done_item:
        return _Verdict("incorrect", raw, _texts(spec, letters))
    want, got = set(tgt), set(letters)  # type: ignore[arg-type]
    extra = {"hits": len(want & got), "misses": len(want - got), "extras": len(got - want)}
    return _Verdict(ok(want == got), raw, _texts(spec, letters), extra=extra)


def _target_text(spec: FieldSpec, tgt: Any) -> Any:
    if spec.answer_type in ("single_choice", "multi_choice") and tgt != NOT_VISIBLE:
        return _texts(spec, [tgt] if isinstance(tgt, str) else list(tgt))
    return None


def score_field(
    spec: FieldSpec,
    target: FieldTarget,
    call_id: str,
    parsed: ParsedResponse | None,
    not_executed_reason: str | None = None,
    *,
    config: ScoringConfig | None = None,
    judge: Judge | None = None,
) -> FieldScore:
    texts = _target_text(spec, target.target)
    one_letter = spec.answer_type == "single_choice" and isinstance(target.target, str)
    # Every score carries these, also when no answer was parsed, so per-type and not-done
    # denominators count every selected field.
    base: dict[str, Any] = {"answer_type": spec.answer_type}
    if spec.allow_not_visible:
        base["not_done_item"] = target.target == NOT_VISIBLE
    common: dict[str, Any] = {
        "report_id": target.report_id,
        "field_id": spec.id,
        "call_id": call_id,
        "target": target.target,
        "target_text": texts[0] if one_letter and texts else texts,
    }
    if parsed is None:
        return FieldScore(
            **common,
            outcome="not_executed",
            correct=False,
            predicted=None,
            predicted_text=None,
            detail=not_executed_reason,
            extra=base,
        )
    if parsed.status != "ok":
        return FieldScore(
            **common,
            outcome="invalid_output",
            correct=False,
            predicted=None,
            predicted_text=None,
            detail=f"{parsed.status}: {parsed.detail}",
            extra=base,
        )
    if spec.id not in parsed.answers:
        return FieldScore(
            **common,
            outcome="missing_answer",
            correct=False,
            predicted=None,
            predicted_text=None,
            detail="field key absent from response",
            extra=base,
        )
    v = _check(spec, target, parsed.answers[spec.id], config or ScoringConfig(), judge)
    return FieldScore(
        **common,
        outcome=v.outcome,
        correct=v.outcome == "correct",
        predicted=v.predicted,
        predicted_text=v.predicted_text,
        detail=v.detail,
        extra={**base, **v.extra},
    )
