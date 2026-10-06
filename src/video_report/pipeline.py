"""Create, resume, and inspect runs. Thin glue between snapshot, run store, provider, runner."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from video_report.config import ExperimentConfig
from video_report.providers import build_provider, preflight
from video_report.providers.base import Provider
from video_report.run_store import RunDir, RunError
from video_report.runner import ExecutionPolicy, Runner, SessionResult
from video_report.snapshot import build_snapshot, code_revision, environment_info

ProviderFactory = Callable[[ExperimentConfig, RunDir], Provider]


class IncompatibleResumeError(RunError):
    pass


def default_provider_factory(cfg: ExperimentConfig, run: RunDir) -> Provider:
    # The mock reads its canned answers from the frozen snapshot copy, never the source file.
    return build_provider(cfg.provider, run.mock_responses_path())


def new_run_id(experiment: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment}-{stamp}-{secrets.token_hex(3)}"


def create_run(
    config_path: Path,
    runs_dir: Path,
    *,
    run_id: str | None = None,
    video_root: str | None = None,
    environ: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> RunDir:
    snap = build_snapshot(config_path, video_root_override=video_root, environ=environ)
    cfg = snap.config
    # Fail before writing anything: invalid overrides, missing SDK or credentials.
    ExecutionPolicy.from_config(cfg.execution, **(overrides or {}))
    preflight(cfg.provider, environ)
    rid = run_id or new_run_id(cfg.name)
    meta: dict[str, Any] = {
        "run_id": rid,
        "experiment": cfg.name,
        "config_path": str(snap.config_path),
        "method": cfg.method.name,
        "provider": cfg.provider.name,
        "model": cfg.provider.model,
        "generation": cfg.generation.model_dump(mode="json"),
        "video": cfg.video.model_dump(mode="json"),
        "video_root": snap.video_root,
        "video_root_source": snap.video_root_source,
        "num_reports": len(snap.reports),
        "num_fields": sum(len(r.fields) for r in snap.reports),
        "num_calls": len(snap.plan),
    }
    return RunDir.create(
        runs_dir / rid,
        meta=meta,
        config=cfg,
        config_source_text=snap.config_source_text,
        prompts=snap.prompts,
        reports=snap.reports,
        plan=snap.plan,
        requests=snap.requests,
        mock_responses_text=snap.mock_responses_text,
        environment=environment_info(),
    )


def check_resume_compatibility(
    run: RunDir,
    *,
    config_path: Path | None,
    video_root: str | None,
    allow_code_change: bool,
    environ: dict[str, str] | None = None,
) -> list[str]:
    """Raise ``IncompatibleResumeError`` on mismatch; return non-fatal notes."""
    notes: list[str] = []
    recorded = run.verify_snapshot()
    problems: list[str] = []

    # The executed package source must match; commit/dirty state is informational.
    frozen_code = run.load_environment().get("code", {})
    current_code = code_revision()
    frozen_src, current_src = frozen_code.get("source_sha256"), current_code.get("source_sha256")
    if frozen_src is None or frozen_src != current_src:
        msg = (
            f"package source changed or unknown: frozen source_sha256={frozen_src} "
            f"(commit {frozen_code.get('commit')}, dirty={frozen_code.get('dirty')}) vs "
            f"current {current_src} (commit {current_code.get('commit')}, "
            f"dirty={current_code.get('dirty')})"
        )
        if allow_code_change:
            notes.append(msg + " [allowed by --allow-code-change]")
        else:
            problems.append(msg + "; pass --allow-code-change to resume anyway")
    elif (frozen_code.get("commit"), frozen_code.get("diff_sha256")) != (
        current_code.get("commit"),
        current_code.get("diff_sha256"),
    ):
        notes.append(
            "repository revision changed but the package source is identical "
            f"(frozen commit {frozen_code.get('commit')}, current {current_code.get('commit')})"
        )

    if config_path is not None:
        snap = build_snapshot(config_path, video_root_override=video_root, environ=environ)
        if snap.config.name != run.read_meta()["experiment"]:
            problems.append(
                f"config is for experiment {snap.config.name!r}, run is "
                f"{run.read_meta()['experiment']!r}"
            )
        changed = sorted(
            k for k in recorded if k != "overall" and snap.fingerprints.get(k) != recorded[k]
        )
        if changed:
            problems.append(
                f"current sources differ from the frozen run in: {changed}. Start a new run "
                "instead (the frozen run keeps its original inputs)"
            )
    if problems:
        raise IncompatibleResumeError("cannot resume: " + " | ".join(problems))
    return notes


def execute_run(
    run: RunDir,
    *,
    mode: str,
    overrides: dict[str, Any] | None = None,
    retry_failed: bool = False,
    provider_factory: ProviderFactory = default_provider_factory,
    on_call_done: Callable[[dict[str, Any]], None] | None = None,
    session_notes: list[str] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> SessionResult:
    cfg = run.load_config()  # always the frozen config
    policy = ExecutionPolicy.from_config(cfg.execution, **(overrides or {}))
    provider = provider_factory(cfg, run)
    try:
        info = provider.describe()
        meta = run.read_meta()
        frozen_info = meta.get("provider_info")
        if frozen_info is not None and frozen_info != info:
            raise IncompatibleResumeError(
                f"provider description changed since run creation: {frozen_info} -> {info}"
            )
        runner = Runner(
            run,
            provider,
            policy,
            mode=mode,
            retry_failed=retry_failed,
            sleep=sleep,
            on_call_done=on_call_done,
            session_info={
                "overrides": {k: v for k, v in (overrides or {}).items() if v is not None},
                "notes": session_notes or [],
                "provider_info": info,
                "code": code_revision(),
            },
        )
        if frozen_info is None:
            # Recorded under the lock on the first session.
            with run.writer_lock():
                meta = run.read_meta()
                meta["provider_info"] = info
                run.write_meta(meta)
        return runner.execute()
    finally:
        provider.close()


def run_status(run: RunDir) -> dict[str, Any]:
    meta = run.read_meta()
    calls = run.read_all_calls()
    by_status: dict[str, int] = {}
    attempts: dict[str, int] = {}
    for c in calls:
        by_status[c["status"]] = by_status.get(c["status"], 0) + 1
        for a in c["attempts"]:
            attempts[a["outcome"]] = attempts.get(a["outcome"], 0) + 1
    state = meta["state"]
    active = run.lock_is_held()
    if state == "running" and not active:
        state = "running (stale: no active writer; the previous process likely crashed — resume)"
    return {
        "run_id": meta["run_id"],
        "experiment": meta["experiment"],
        "state": state,
        "active_writer": active,
        "calls": by_status,
        "attempt_outcomes": attempts,
        "sessions": len(meta["sessions"]),
    }
