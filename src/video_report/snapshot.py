"""Build the frozen inputs of a run from an experiment config and its source files.

Everything a run needs is resolved and rendered here, once, at creation time. Execution later
reads only the frozen copy in the run directory, so editing prompts, reports or the config while
a run is in progress cannot change it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_report.benchmarks.schema import Report
from video_report.config import ExperimentConfig, load_config, resolve_relative
from video_report.datasets.base import (
    DataValidationError,
    ResolvedVideo,
    choose_video_root,
    load_benchmark,
    load_reports,
    missing_local_videos,
    resolve_video,
)
from video_report.methods import METHODS, CallSpec, ReportSelection, check_plan
from video_report.prompting import PromptTemplates, render
from video_report.providers import validate_request_for
from video_report.providers.base import ModelRequest, VideoInput
from video_report.providers.mock import load_mock_responses
from video_report.run_store import snapshot_fingerprints


@dataclass(frozen=True)
class Snapshot:
    config_path: Path
    config: ExperimentConfig
    config_source_text: str
    prompts: dict[str, str]
    reports: list[Report]  # selected reports, restricted to selected fields
    plan: list[CallSpec]
    requests: list[ModelRequest]
    mock_responses_text: str | None
    video_root: str | None
    video_root_source: str
    fingerprints: dict[str, str]


def select_reports(reports: list[Report], cfg: ExperimentConfig) -> list[Report]:
    by_id = {r.id: r for r in reports}
    if cfg.data.report_ids is not None:
        unknown = [i for i in cfg.data.report_ids if i not in by_id]
        if unknown:
            raise DataValidationError(f"data.report_ids not found in reports: {unknown}")
        if len(set(cfg.data.report_ids)) != len(cfg.data.report_ids):
            raise DataValidationError("data.report_ids contains duplicates")
        wanted = set(cfg.data.report_ids)
        reports = [r for r in reports if r.id in wanted]
    if cfg.data.field_ids is None:
        return reports
    if len(set(cfg.data.field_ids)) != len(cfg.data.field_ids):
        raise DataValidationError("data.field_ids contains duplicates")
    out = []
    for r in reports:
        have = {f.id for f in r.fields}
        missing = [f for f in cfg.data.field_ids if f not in have]
        if missing:
            raise DataValidationError(f"report {r.id!r} lacks selected fields {missing}")
        keep = [f for f in r.fields if f.id in set(cfg.data.field_ids)]
        out.append(r.model_copy(update={"fields": keep}))
    return out


def build_snapshot(
    config_path: Path,
    *,
    video_root_override: str | None = None,
    environ: dict[str, str] | None = None,
) -> Snapshot:
    config_path = config_path.resolve()
    cfg, source_text = load_config(config_path)
    base = config_path.parent

    benchmark = load_benchmark(resolve_relative(base, cfg.data.benchmark))
    reports = select_reports(load_reports(benchmark.reports_path), cfg)

    prompts = {"user.txt": resolve_relative(base, cfg.prompt.user).read_text(encoding="utf-8")}
    if cfg.prompt.system:
        prompts["system.txt"] = resolve_relative(base, cfg.prompt.system).read_text(
            encoding="utf-8"
        )
    templates = PromptTemplates(prompts.get("system.txt"), prompts["user.txt"])
    templates.check()

    selections = [ReportSelection(r.id, tuple(f.id for f in r.fields)) for r in reports]
    plan = METHODS[cfg.method.name](selections)
    check_plan(plan, selections)

    config_root = resolve_relative(base, cfg.data.video_root) if cfg.data.video_root else None
    video_root, root_source = choose_video_root(
        video_root_override, config_root, benchmark, environ
    )
    resolved: dict[str, list[ResolvedVideo]] = {
        r.id: [resolve_video(v.id, v.uri, v.mime_type, video_root) for v in r.input.videos]
        for r in reports
    }
    if cfg.data.require_local_videos:
        missing = [m for vids in resolved.values() for m in missing_local_videos(vids)]
        if missing:
            raise DataValidationError(f"missing local video files: {missing}")

    by_id = {r.id: r for r in reports}
    requests = []
    for call in plan:
        report = by_id[call.report_id]
        fields = [report.field(f) for f in call.field_ids]
        system, user = render(templates, fields, report.input.context, cfg.prompt.context_keys)
        videos = []
        for ref, rv in zip(report.input.videos, resolved[report.id], strict=True):
            tr = ref.time_range if cfg.video.use_time_range else None
            videos.append(
                VideoInput(
                    video_id=rv.video_id,
                    location=rv.location,
                    is_local=rv.is_local,
                    mime_type=rv.mime_type,
                    start_s=tr.start if tr else None,
                    end_s=tr.end if tr else None,
                )
            )
        req = ModelRequest(
            call_id=call.call_id,
            report_id=call.report_id,
            field_ids=call.field_ids,
            system=system,
            user=user,
            videos=tuple(videos),
            model=cfg.provider.model,
            generation=cfg.generation.model_dump(mode="json"),
            fps=cfg.video.fps,
        )
        try:
            validate_request_for(cfg.provider.name, req)
        except ValueError as exc:
            raise DataValidationError(str(exc)) from exc
        requests.append(req)

    mock_text = None
    if cfg.provider.name == "mock":
        if cfg.provider.mock.responses:
            mock_path: Path | None = resolve_relative(base, cfg.provider.mock.responses)
        else:
            mock_path = benchmark.mock_responses_path
        if mock_path is None:
            raise DataValidationError("mock provider: no responses file configured")
        mock_text = mock_path.read_text(encoding="utf-8")
        load_mock_responses(mock_path)  # validate format now, not mid-run

    fps = snapshot_fingerprints(cfg, prompts, reports, plan, requests, mock_text)
    return Snapshot(
        config_path=config_path,
        config=cfg,
        config_source_text=source_text,
        prompts=prompts,
        reports=reports,
        plan=plan,
        requests=requests,
        mock_responses_text=mock_text,
        video_root=str(video_root) if video_root else None,
        video_root_source=root_source,
        fingerprints=fps,
    )


# --------------------------------------------------------------------------- environment


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def package_source_sha256() -> str:
    """Hash of every .py file of this package (path + content): the code that actually runs.

    Works with or without git, and covers untracked files.
    """
    pkg = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for f in sorted(pkg.rglob("*.py")):
        digest.update(str(f.relative_to(pkg)).encode() + b"\0")
        digest.update(f.read_bytes() + b"\0")
    return digest.hexdigest()


def code_revision(repo_dir: Path | None = None) -> dict[str, Any]:
    """Commit, dirty flag, hash of uncommitted changes (incl. untracked files), source hash.

    Diff contents are hashed, never stored.
    """
    info: dict[str, Any] = {"source_sha256": package_source_sha256()}
    repo = repo_dir or Path(__file__).resolve().parents[2]
    commit = _git(["rev-parse", "HEAD"], repo)
    if commit is None:
        return {
            **info,
            "commit": None,
            "dirty": None,
            "diff_sha256": None,
            "note": "not a git checkout",
        }
    status = _git(["status", "--porcelain"], repo) or ""
    digest = hashlib.sha256((_git(["diff", "HEAD"], repo) or "").encode())
    untracked = _git(["ls-files", "--others", "--exclude-standard", "-z"], repo) or ""
    for rel in sorted(x for x in untracked.split("\0") if x):
        path = repo / rel
        digest.update(rel.encode() + b"\0")
        if path.is_file():
            digest.update(path.read_bytes() + b"\0")
    dirty = bool(status.strip())
    return {
        **info,
        "commit": commit.strip(),
        "dirty": dirty,
        "diff_sha256": digest.hexdigest() if dirty else None,
    }


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_info() -> dict[str, Any]:
    return {
        "code": code_revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            d: _version(d) for d in ("video-report", "pydantic", "PyYAML", "google-genai")
        },
    }
