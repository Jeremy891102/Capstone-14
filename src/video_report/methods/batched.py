"""Split each report's frozen order into fixed-size, independent calls."""

from collections.abc import Sequence

from video_report.methods.base import CallSpec, ReportSelection


def plan_batched(selections: Sequence[ReportSelection], batch_size: int) -> list[CallSpec]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return [
        CallSpec(
            f"{s.report_id}::batch={i // batch_size + 1}",
            s.report_id,
            s.field_ids[i : i + batch_size],
        )
        for s in selections
        for i in range(0, len(s.field_ids), batch_size)
    ]
