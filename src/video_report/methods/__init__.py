"""Call-planning methods. Adding a method = adding a function and one entry in ``METHODS``."""

from __future__ import annotations

from video_report.methods.base import CallSpec, Planner, ReportSelection, check_plan
from video_report.methods.per_field import plan_per_field
from video_report.methods.whole_report import plan_whole_report

METHODS: dict[str, Planner] = {
    "whole_report": plan_whole_report,
    "per_field": plan_per_field,
}

__all__ = ["METHODS", "CallSpec", "Planner", "ReportSelection", "check_plan"]
