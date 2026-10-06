"""Readers and validators for prepared benchmark data (data contract ``video_report.v1``).

Teammate integration: dataset preparation (HD-EPIC, CaptainCook4D, ...) happens outside this
repository and must emit files matching ``docs/data_contract.md``. If an intermediate format is
easier, write a small function that yields plain dicts and pass them through ``parse_report`` /
``parse_target`` so the same validation applies. No dataset-specific conversion lives here.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from video_report.benchmarks.schema import (
    BenchmarkManifest,
    FieldTarget,
    Report,
    letter_index,
)
from video_report.io_utils import iter_jsonl, read_json

VIDEO_ROOT_ENV = "VIDEO_REPORT_VIDEO_ROOT"

_MIME_BY_SUFFIX = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
}


class DataValidationError(ValueError):
    """Prepared data violates the data contract."""


def _format_validation_error(where: str, exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return f"{where}: " + "; ".join(parts)


def parse_report(obj: Any, where: str = "<report>") -> Report:
    try:
        return Report.model_validate(obj)
    except ValidationError as exc:
        raise DataValidationError(_format_validation_error(where, exc)) from exc


def parse_target(obj: Any, where: str = "<target>") -> FieldTarget:
    try:
        return FieldTarget.model_validate(obj)
    except ValidationError as exc:
        raise DataValidationError(_format_validation_error(where, exc)) from exc


def check_unique_report_ids(reports: Sequence[Report]) -> None:
    seen: set[str] = set()
    for r in reports:
        if r.id in seen:
            raise DataValidationError(f"duplicate report id: {r.id!r}")
        seen.add(r.id)


def load_reports(path: Path) -> list[Report]:
    try:
        rows = [parse_report(obj, f"{path}:{lineno}") for lineno, obj in iter_jsonl(path)]
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc
    if not rows:
        raise DataValidationError(f"{path}: no reports found")
    check_unique_report_ids(rows)
    return rows


def load_ground_truth(path: Path) -> list[FieldTarget]:
    try:
        rows = [parse_target(obj, f"{path}:{lineno}") for lineno, obj in iter_jsonl(path)]
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc
    seen: set[tuple[str, str]] = set()
    for t in rows:
        key = (t.report_id, t.field_id)
        if key in seen:
            raise DataValidationError(f"{path}: duplicate target for {key}")
        seen.add(key)
    return rows


def validate_targets(
    reports: Iterable[Report],
    targets: Iterable[FieldTarget],
    selection: dict[str, list[str]],
    *,
    reject_unknown_fields: bool = True,
) -> dict[tuple[str, str], FieldTarget]:
    """Check that every selected (report, field) has exactly one valid target.

    ``selection`` maps report id -> selected field ids. Targets for reports/fields outside
    the selection are ignored (a run may cover a subset). With ``reject_unknown_fields``, a
    target naming a field that the report does not have is an error; evaluation passes
    False because a run's frozen reports only contain the selected fields.
    """
    by_id = {r.id: r for r in reports}
    index: dict[tuple[str, str], FieldTarget] = {}
    for t in targets:
        if t.report_id not in selection:
            continue
        report = by_id.get(t.report_id)
        if report is None:
            raise DataValidationError(f"target references unknown report {t.report_id!r}")
        if t.field_id not in selection[t.report_id]:
            if reject_unknown_fields and t.field_id not in {f.id for f in report.fields}:
                raise DataValidationError(
                    f"target references unknown field {t.field_id!r} of report {t.report_id!r}"
                )
            continue
        spec = report.field(t.field_id)
        idx = letter_index(t.target)
        if idx is None or idx >= len(spec.choices):
            raise DataValidationError(
                f"target for ({t.report_id}, {t.field_id}) is {t.target!r}; expected one of "
                f"{spec.letters()}"
            )
        if t.target_text is not None and t.target_text != spec.choices[idx]:
            raise DataValidationError(
                f"target_text mismatch for ({t.report_id}, {t.field_id}): letter {t.target} is "
                f"{spec.choices[idx]!r} but target_text is {t.target_text!r}"
            )
        index[(t.report_id, t.field_id)] = t
    missing = [
        (rid, fid) for rid, fids in selection.items() for fid in fids if (rid, fid) not in index
    ]
    if missing:
        raise DataValidationError(f"missing targets for selected fields: {missing}")
    return index


# --------------------------------------------------------------------------- benchmark dirs


@dataclass(frozen=True)
class Benchmark:
    root: Path
    manifest: BenchmarkManifest

    @property
    def reports_path(self) -> Path:
        return self.root / self.manifest.reports_file

    @property
    def ground_truth_path(self) -> Path | None:
        gt = self.manifest.ground_truth_file
        return self.root / gt if gt else None

    @property
    def mock_responses_path(self) -> Path | None:
        mr = self.manifest.mock_responses_file
        return self.root / mr if mr else None


def load_benchmark(root: Path) -> Benchmark:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise DataValidationError(f"benchmark manifest not found: {manifest_path}")
    try:
        manifest = BenchmarkManifest.model_validate(read_json(manifest_path))
    except ValidationError as exc:
        raise DataValidationError(_format_validation_error(str(manifest_path), exc)) from exc
    return Benchmark(root=root.resolve(), manifest=manifest)


# --------------------------------------------------------------------------- video paths


@dataclass(frozen=True)
class ResolvedVideo:
    video_id: str
    uri: str  # as written in reports.jsonl
    location: str  # absolute local path, or the URI unchanged when remote
    is_local: bool
    mime_type: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "uri": self.uri,
            "location": self.location,
            "is_local": self.is_local,
            "mime_type": self.mime_type,
        }


def choose_video_root(
    cli_root: str | None,
    config_root: Path | None,
    benchmark: Benchmark | None,
    env: dict[str, str] | None = None,
) -> tuple[Path | None, str]:
    """Pick the video root. Precedence (first set wins):

    1. ``--video-root`` on the command line
    2. ``$VIDEO_REPORT_VIDEO_ROOT``
    3. ``data.video_root`` in the experiment config (relative to the config file)
    4. ``video_root`` in the benchmark manifest (relative to the manifest)
    5. the benchmark directory itself
    """
    environ = os.environ if env is None else env
    if cli_root:
        return Path(cli_root).expanduser().resolve(), "cli"
    env_root = environ.get(VIDEO_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve(), "env"
    if config_root is not None:
        return config_root.resolve(), "config"
    if benchmark is not None:
        if benchmark.manifest.video_root:
            return (benchmark.root / benchmark.manifest.video_root).resolve(), "manifest"
        return benchmark.root, "benchmark_dir"
    return None, "none"


def resolve_video(
    video_id: str, uri: str, mime_type: str | None, video_root: Path | None
) -> ResolvedVideo:
    parsed = urlparse(uri)
    # A one-letter "scheme" is a Windows drive letter, not a URI scheme.
    if parsed.scheme and len(parsed.scheme) > 1:
        if parsed.scheme == "file":
            location = str(Path(parsed.path).resolve())
            is_local = True
        else:
            location, is_local = uri, False
    else:
        path = Path(uri).expanduser()
        if not path.is_absolute():
            if video_root is None:
                raise DataValidationError(
                    f"video {video_id!r} has relative uri {uri!r} but no video root is configured"
                )
            path = video_root / path
        location, is_local = str(path.resolve()), True
    if mime_type is None and is_local:
        mime_type = _MIME_BY_SUFFIX.get(Path(location).suffix.lower())
    return ResolvedVideo(video_id, uri, location, is_local, mime_type)


def missing_local_videos(videos: Iterable[ResolvedVideo]) -> list[str]:
    return [v.location for v in videos if v.is_local and not Path(v.location).is_file()]
