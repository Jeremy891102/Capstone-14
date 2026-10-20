"""Preflight checks for the fixed two-report HD-EPIC pilot."""

from __future__ import annotations

import math
from pathlib import Path

from video_report.datasets.base import load_ground_truth, load_reports, validate_targets
from video_report.io_utils import sha256_file


def validate_pilot_budget(calls: int, estimate: float, limit: float) -> None:
    if calls != 124:
        raise ValueError("pilot scope must contain exactly 124 calls")
    if not (math.isfinite(estimate) and math.isfinite(limit) and 0 <= estimate <= limit):
        raise ValueError("pilot estimate exceeds budget or budget is invalid")


def verify_derived_ground_truth(source: Path, derived: Path, reports_path: Path) -> str:
    reports = load_reports(reports_path)
    targets = load_ground_truth(derived)
    selection = {r.id: [f.id for f in r.fields] for r in reports}
    checked = validate_targets(reports, targets, selection)
    originals = {(t.report_id, t.field_id): t for t in load_ground_truth(source)}
    if len(checked) != len(targets):
        raise ValueError("derived ground truth contains fields outside the selected reports")
    for key, target in checked.items():
        original = originals.get(key)
        if original is None or (target.target, target.target_text) != (
            original.target,
            original.target_text,
        ):
            raise ValueError(f"derived ground truth disagrees with source for {key}")
    return sha256_file(derived)
