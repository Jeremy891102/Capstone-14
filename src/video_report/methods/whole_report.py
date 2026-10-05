"""One model call per report, answering every selected field."""

from __future__ import annotations

from collections.abc import Sequence

from video_report.methods.base import CallSpec, ReportSelection


def plan_whole_report(selections: Sequence[ReportSelection]) -> list[CallSpec]:
    return [CallSpec(f"{s.report_id}::whole", s.report_id, s.field_ids) for s in selections]
