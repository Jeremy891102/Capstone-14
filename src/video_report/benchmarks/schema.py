"""Data contract v1 (``video_report.v1``): model-facing reports, ground truth, benchmark manifest.

Naming follows common evaluation conventions (``id``, ``input``, ``choices``, ``target``,
``metadata``, as used by e.g. Inspect AI's ``Sample``) wrapped in a small project-specific
report structure. This schema is our own; it is not an industry standard.

Separation rule: ``Report`` (reports.jsonl) is everything inference may see. ``FieldTarget``
(ground_truth.jsonl) holds targets and privileged evidence and is only read by evaluation.
``Report`` forbids unknown keys, so a stray ``target``/``answer`` key in reports.jsonl is a
validation error rather than a silent leak.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "video_report.v1"

# v1 supports exactly one answer type. Anything else is rejected at load time instead of
# being scored with the wrong rule.
SUPPORTED_ANSWER_TYPES: frozenset[str] = frozenset({"single_choice"})

MAX_CHOICES = 26  # one uppercase letter per choice

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _check_id(value: str) -> str:
    if not _ID_RE.match(value):
        raise ValueError(
            f"invalid id {value!r}: use 1-128 chars of [A-Za-z0-9._-], starting alphanumeric"
        )
    return value


Id = Annotated[str, AfterValidator(_check_id)]
ContextValue = str | int | float | bool


def choice_letter(index: int) -> str:
    if not 0 <= index < MAX_CHOICES:
        raise ValueError(f"choice index out of range: {index}")
    return chr(ord("A") + index)


def letter_index(letter: str) -> int | None:
    """Return the 0-based index for a single uppercase letter, else None."""
    if len(letter) == 1 and "A" <= letter <= "Z":
        return ord(letter) - ord("A")
    return None


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TimeRange(StrictModel):
    """A half-open interval [start, end) on the referenced video's own timeline.

    ``unit``: only ``seconds`` in v1. ``reference``: ``video_start`` means offsets are measured
    from the first frame of the referenced file (presentation time 0), not from any dataset-wide
    or wall-clock time base. Dataset converters must translate other conventions.
    """

    start: float = Field(ge=0)
    end: float
    unit: Literal["seconds"]
    reference: Literal["video_start"]

    @model_validator(mode="after")
    def _check_order(self) -> TimeRange:
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise ValueError("time_range start/end must be finite numbers")
        if self.end <= self.start:
            raise ValueError(f"time_range end ({self.end}) must be > start ({self.start})")
        return self


class VideoRef(StrictModel):
    id: Id
    # Relative path (resolved against the video root), absolute path, or URI with a scheme.
    uri: str = Field(min_length=1)
    mime_type: str | None = None
    time_range: TimeRange | None = None


class ReportInput(StrictModel):
    videos: list[VideoRef] = Field(min_length=1)
    # Candidate model-facing context. Only keys whitelisted by the experiment's
    # ``prompt.context_keys`` are ever rendered into a prompt.
    context: dict[str, ContextValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_video_ids(self) -> ReportInput:
        ids = [v.id for v in self.videos]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate video ids: {dupes}")
        return self


class FieldSpec(StrictModel):
    id: Id
    question: str = Field(min_length=1)
    answer_type: str
    # Ordered. Choice i is presented to the model as letter chr(65 + i).
    choices: list[str]

    @model_validator(mode="after")
    def _check(self) -> FieldSpec:
        if self.answer_type not in SUPPORTED_ANSWER_TYPES:
            raise ValueError(
                f"field {self.id!r}: unsupported answer_type {self.answer_type!r}; "
                f"v1 supports only {sorted(SUPPORTED_ANSWER_TYPES)}"
            )
        if not 2 <= len(self.choices) <= MAX_CHOICES:
            raise ValueError(
                f"field {self.id!r}: single_choice needs 2..{MAX_CHOICES} choices, "
                f"got {len(self.choices)}"
            )
        normalized = [c.strip() for c in self.choices]
        if any(not c for c in normalized):
            raise ValueError(f"field {self.id!r}: choices must be non-empty strings")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"field {self.id!r}: duplicate choices {self.choices}")
        return self

    def letters(self) -> list[str]:
        return [choice_letter(i) for i in range(len(self.choices))]


class SourceRef(StrictModel):
    """Provenance. Never sent to the model."""

    dataset: str = Field(min_length=1)
    split: str | None = None
    original_ids: dict[str, str] = Field(default_factory=dict)
    # Pointers into the original annotations, e.g. "annotations/recipes.csv#row=17".
    annotation_refs: list[str] = Field(default_factory=list)


class Report(StrictModel):
    id: Id
    input: ReportInput
    fields: list[FieldSpec] = Field(min_length=1)
    source: SourceRef
    # Free-form bookkeeping. Never sent to the model.
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_field_ids(self) -> Report:
        ids = [f.id for f in self.fields]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"report {self.id!r}: duplicate field ids {dupes}")
        return self

    def field(self, field_id: str) -> FieldSpec:
        for f in self.fields:
            if f.id == field_id:
                return f
        raise KeyError(f"report {self.id!r} has no field {field_id!r}")


class FieldTarget(StrictModel):
    """One ground-truth row: the target for one (report, field). Evaluation-only."""

    report_id: Id
    field_id: Id
    # The correct choice letter (Inspect-style multiple-choice target).
    target: str
    # Optional cross-check: if given, must equal choices[target] exactly. Catches choice
    # reordering between reports.jsonl and ground_truth.jsonl.
    target_text: str | None = None
    # Privileged answer evidence (annotation text, timestamps, annotator ids, ...).
    evidence: dict[str, Any] = Field(default_factory=dict)


class BenchmarkManifest(StrictModel):
    schema_version: Literal["video_report.v1"]
    benchmark_id: Id
    description: str = ""
    reports_file: str = "reports.jsonl"
    ground_truth_file: str | None = "ground_truth.jsonl"
    mock_responses_file: str | None = None
    # Default root for relative video URIs, relative to the manifest's directory.
    video_root: str | None = None
    source_datasets: list[str] = Field(default_factory=list)
    notes: str = ""
