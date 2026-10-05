"""Call planning: a method turns selected reports into a list of model calls.

A method decides *which* fields go into *which* call. It never sees targets or evidence: its
input is ``ReportSelection`` (report ids + selected field ids, in order).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReportSelection:
    report_id: str
    field_ids: tuple[str, ...]


@dataclass(frozen=True)
class CallSpec:
    """One planned model call. ``call_id`` is stable across runs given the same selection."""

    call_id: str
    report_id: str
    field_ids: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "report_id": self.report_id,
            "field_ids": list(self.field_ids),
        }

    @staticmethod
    def from_json(obj: dict[str, Any]) -> CallSpec:
        return CallSpec(obj["call_id"], obj["report_id"], tuple(obj["field_ids"]))


Planner = Callable[[Sequence[ReportSelection]], list[CallSpec]]


def check_plan(plan: Sequence[CallSpec], selections: Sequence[ReportSelection]) -> None:
    """Every selected (report, field) must be asked exactly once; call ids must be unique."""
    ids = [c.call_id for c in plan]
    if len(set(ids)) != len(ids):
        raise ValueError("call plan has duplicate call ids")
    asked: dict[tuple[str, str], int] = {}
    for c in plan:
        if not c.field_ids:
            raise ValueError(f"call {c.call_id!r} has no fields")
        for f in c.field_ids:
            asked[(c.report_id, f)] = asked.get((c.report_id, f), 0) + 1
    expected = {(s.report_id, f) for s in selections for f in s.field_ids}
    if set(asked) != expected or any(n != 1 for n in asked.values()):
        raise ValueError("call plan does not cover each selected field exactly once")
