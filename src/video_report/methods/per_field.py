"""One model call per (report, field)."""

from __future__ import annotations

from collections.abc import Sequence

from video_report.methods.base import CallSpec, ReportSelection


def plan_per_field(selections: Sequence[ReportSelection]) -> list[CallSpec]:
    return [
        CallSpec(f"{s.report_id}::field={fid}", s.report_id, (fid,))
        for s in selections
        for fid in s.field_ids
    ]
