"""Score one field: compare a parsed answer with its target. Pure function; no I/O.

Outcomes (every selected field gets exactly one; all count in the accuracy denominator):

* ``correct``          valid letter equal to the target
* ``incorrect``        valid letter, not the target
* ``invalid_answer``   key present but value is not one of this field's letters
* ``missing_answer``   response parsed but this field's key is absent
* ``invalid_output``   whole response unusable (empty, invalid JSON, duplicate keys, not object)
* ``not_executed``     no successful provider call for this field (failed / pending / interrupted)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from video_report.benchmarks.schema import FieldSpec, FieldTarget, letter_index
from video_report.evaluation.parser import ParsedResponse

SCORER_VERSION = "mcq_exact_letter_v1"

Outcome = Literal[
    "correct", "incorrect", "invalid_answer", "missing_answer", "invalid_output", "not_executed"
]
VALID_ANSWER_OUTCOMES = frozenset({"correct", "incorrect"})


@dataclass(frozen=True)
class FieldScore:
    report_id: str
    field_id: str
    call_id: str
    outcome: Outcome
    correct: bool
    predicted: str | None  # the valid letter, if any
    predicted_text: str | None
    target: str
    target_text: str
    detail: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def score_field(
    spec: FieldSpec,
    target: FieldTarget,
    call_id: str,
    parsed: ParsedResponse | None,
    not_executed_reason: str | None = None,
) -> FieldScore:
    tidx = letter_index(target.target)
    assert tidx is not None and tidx < len(spec.choices), "targets are validated before scoring"
    common = {
        "report_id": target.report_id,
        "field_id": spec.id,
        "call_id": call_id,
        "target": target.target,
        "target_text": spec.choices[tidx],
    }
    if parsed is None:
        return FieldScore(
            **common,
            outcome="not_executed",
            correct=False,
            predicted=None,
            predicted_text=None,
            detail=not_executed_reason,
        )
    if parsed.status != "ok":
        return FieldScore(
            **common,
            outcome="invalid_output",
            correct=False,
            predicted=None,
            predicted_text=None,
            detail=f"{parsed.status}: {parsed.detail}",
        )
    if spec.id not in parsed.answers:
        return FieldScore(
            **common,
            outcome="missing_answer",
            correct=False,
            predicted=None,
            predicted_text=None,
            detail="field key absent from response",
        )
    raw = parsed.answers[spec.id]
    idx = letter_index(raw) if isinstance(raw, str) else None
    if idx is None or idx >= len(spec.choices):
        return FieldScore(
            **common,
            outcome="invalid_answer",
            correct=False,
            predicted=None,
            predicted_text=None,
            detail=f"value {raw!r} is not one of {spec.letters()}",
        )
    correct = idx == tidx
    return FieldScore(
        **common,
        outcome="correct" if correct else "incorrect",
        correct=correct,
        predicted=raw,
        predicted_text=spec.choices[idx],
    )
