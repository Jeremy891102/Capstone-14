"""Run directory layout, writer lock, and per-call checkpoint persistence.

Layout of ``runs/<run_id>/``::

    run.json                  run metadata + state (atomic rewrite)
    run.lock                  flock() target; held by the single active writer
    snapshot/                 frozen inputs, written once at creation, never modified
      config.json             resolved config        config.source.yaml   original text
      prompts/{system,user}.txt
      reports.jsonl           selected model-facing reports (selected fields only)
      call_plan.jsonl         requests.jsonl (exact provider inputs)
      mock_responses.jsonl    (mock provider only)
      fingerprints.json       environment.json
    calls/<key>.json          per-call checkpoint: status + every attempt (atomic rewrite)
    responses/<key>/attempt-0001.json   raw provider response, written BEFORE the checkpoint
    predictions.jsonl         derived: one line per succeeded call, rebuilt atomically
    evaluations/<eval_id>/    offline evaluations (never touch the files above)

Lock semantics: ``fcntl.flock(LOCK_EX | LOCK_NB)`` on ``run.lock``. The kernel releases the lock
when the holding process exits for any reason (including SIGKILL), so a crashed writer never
leaves a stale lock. Not supported on Windows or reliable on some network filesystems (NFS).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import socket
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from video_report.benchmarks.schema import Report
from video_report.config import ExperimentConfig
from video_report.io_utils import (
    atomic_write_json,
    atomic_write_text,
    fingerprint,
    iter_jsonl,
    read_json,
    sha256_text,
    utc_now_iso,
    write_jsonl,
)
from video_report.methods.base import CallSpec
from video_report.providers.base import ModelRequest

RUN_FORMAT_VERSION = 1
_LOCK_RETRIES = 20
_LOCK_RETRY_INTERVAL_S = 0.05

RunState = Literal["created", "running", "completed", "completed_with_failures", "interrupted"]
CallStatus = Literal["pending", "in_progress", "succeeded", "failed"]
# Attempt outcomes:
#   in_progress      written before the provider call
#   success          response persisted
#   retryable_error  transient provider failure (request not processed)
#   timeout          no response in time; remote outcome unknown (may be billed)
#   permanent_error  provider rejected the request; not retried
#   interrupted      process died mid-attempt; remote outcome unknown (may be billed)
# A retry interrupted during backoff leaves the last attempt as retryable_error/timeout and
# the call 'pending'.


class RunError(RuntimeError):
    pass


class RunLockedError(RunError):
    pass


def call_key(call_id: str) -> str:
    """Filesystem-safe, collision-free file stem for a call id."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", call_id)[:80]
    return f"{safe}-{hashlib.sha256(call_id.encode()).hexdigest()[:10]}"


def snapshot_fingerprints(
    config: ExperimentConfig,
    prompts: dict[str, str],
    reports: list[Report],
    plan: list[CallSpec],
    requests: list[ModelRequest],
    mock_responses_text: str | None,
) -> dict[str, str]:
    fps = {
        "config": fingerprint(config.inference_view()),
        "prompts": fingerprint(prompts),
        "reports": fingerprint([r.model_dump(mode="json") for r in reports]),
        "call_plan": fingerprint([c.to_json() for c in plan]),
        "requests": fingerprint([r.to_json() for r in requests]),
    }
    if mock_responses_text is not None:
        fps["mock_responses"] = sha256_text(mock_responses_text)
    fps["overall"] = fingerprint(fps)
    return fps


class RunDir:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.run_json = path / "run.json"
        self.lock_path = path / "run.lock"
        self.snapshot = path / "snapshot"
        self.calls_dir = path / "calls"
        self.responses_dir = path / "responses"
        self.predictions_path = path / "predictions.jsonl"
        self.evaluations_dir = path / "evaluations"
        self._lock_fd: int | None = None

    # ----------------------------------------------------------------- creation

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        meta: dict[str, Any],
        config: ExperimentConfig,
        config_source_text: str,
        prompts: dict[str, str],
        reports: list[Report],
        plan: list[CallSpec],
        requests: list[ModelRequest],
        mock_responses_text: str | None,
        environment: dict[str, Any],
    ) -> RunDir:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.mkdir()  # exclusive: two processes can never share a run directory
        except FileExistsError:
            raise RunError(f"run directory already exists: {path}") from None
        run = cls(path)
        snap = run.snapshot
        atomic_write_json(snap / "config.json", config.model_dump(mode="json"))
        atomic_write_text(snap / "config.source.yaml", config_source_text)
        for name, text in prompts.items():
            atomic_write_text(snap / "prompts" / name, text)
        write_jsonl(snap / "reports.jsonl", [r.model_dump(mode="json") for r in reports])
        write_jsonl(snap / "call_plan.jsonl", [c.to_json() for c in plan])
        write_jsonl(snap / "requests.jsonl", [r.to_json() for r in requests])
        if mock_responses_text is not None:
            atomic_write_text(snap / "mock_responses.jsonl", mock_responses_text)
        fps = snapshot_fingerprints(config, prompts, reports, plan, requests, mock_responses_text)
        atomic_write_json(snap / "fingerprints.json", fps)
        atomic_write_json(snap / "environment.json", environment)
        now = utc_now_iso()
        for c in plan:
            run.write_call(
                {
                    "call_id": c.call_id,
                    "report_id": c.report_id,
                    "field_ids": list(c.field_ids),
                    "status": "pending",
                    "attempts": [],
                    "final_attempt": None,
                    "updated_at": now,
                }
            )
        meta = {
            **meta,
            "format_version": RUN_FORMAT_VERSION,
            "created_at": now,
            "state": "created",
            "fingerprints": fps,
            "sessions": [],
        }
        atomic_write_json(run.run_json, meta)
        return run

    # ----------------------------------------------------------------- frozen inputs

    def load_config(self) -> ExperimentConfig:
        return ExperimentConfig.model_validate(read_json(self.snapshot / "config.json"))

    def load_prompts(self) -> dict[str, str]:
        d = self.snapshot / "prompts"
        return {p.name: p.read_text(encoding="utf-8") for p in sorted(d.iterdir()) if p.is_file()}

    def load_reports(self) -> list[Report]:
        return [Report.model_validate(o) for _, o in iter_jsonl(self.snapshot / "reports.jsonl")]

    def load_plan(self) -> list[CallSpec]:
        return [CallSpec.from_json(o) for _, o in iter_jsonl(self.snapshot / "call_plan.jsonl")]

    def load_requests(self) -> dict[str, ModelRequest]:
        reqs = [ModelRequest.from_json(o) for _, o in iter_jsonl(self.snapshot / "requests.jsonl")]
        return {r.call_id: r for r in reqs}

    def mock_responses_path(self) -> Path | None:
        p = self.snapshot / "mock_responses.jsonl"
        return p if p.is_file() else None

    def load_fingerprints(self) -> dict[str, str]:
        fps: dict[str, str] = read_json(self.snapshot / "fingerprints.json")
        return fps

    def load_environment(self) -> dict[str, Any]:
        env: dict[str, Any] = read_json(self.snapshot / "environment.json")
        return env

    def verify_snapshot(self) -> dict[str, str]:
        """Recompute fingerprints from frozen files; raise if anything was modified."""
        mock = self.mock_responses_path()
        current = snapshot_fingerprints(
            self.load_config(),
            self.load_prompts(),
            self.load_reports(),
            self.load_plan(),
            list(self.load_requests().values()),
            mock.read_text(encoding="utf-8") if mock else None,
        )
        recorded = self.load_fingerprints()
        changed = sorted(k for k in recorded if current.get(k) != recorded[k])
        if changed:
            raise RunError(f"frozen snapshot was modified (components: {changed}) in {self.path}")
        return recorded

    # ----------------------------------------------------------------- run metadata

    def read_meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = read_json(self.run_json)
        return meta

    def write_meta(self, meta: dict[str, Any]) -> None:
        self._require_lock()
        atomic_write_json(self.run_json, meta)

    # ----------------------------------------------------------------- calls

    def call_path(self, call_id: str) -> Path:
        return self.calls_dir / f"{call_key(call_id)}.json"

    def read_call(self, call_id: str) -> dict[str, Any]:
        state: dict[str, Any] = read_json(self.call_path(call_id))
        return state

    def write_call(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now_iso()
        atomic_write_json(self.call_path(state["call_id"]), state)

    def read_all_calls(self) -> list[dict[str, Any]]:
        return [self.read_call(c.call_id) for c in self.load_plan()]

    def response_relpath(self, call_id: str, attempt: int) -> str:
        return f"responses/{call_key(call_id)}/attempt-{attempt:04d}.json"

    def write_response(self, call_id: str, attempt: int, payload: dict[str, Any]) -> str:
        rel = self.response_relpath(call_id, attempt)
        atomic_write_json(self.path / rel, payload)
        return rel

    def read_response(self, relpath: str) -> dict[str, Any]:
        payload: dict[str, Any] = read_json(self.path / relpath)
        return payload

    def rebuild_predictions(self) -> int:
        """Write predictions.jsonl from checkpoints: exactly one row per succeeded call."""
        rows = []
        for state in self.read_all_calls():
            if state["status"] != "succeeded":
                continue
            final = state["attempts"][state["final_attempt"] - 1]
            payload = self.read_response(final["response_file"])
            rows.append(
                {
                    "call_id": state["call_id"],
                    "report_id": state["report_id"],
                    "field_ids": state["field_ids"],
                    "attempt": state["final_attempt"],
                    "response_file": final["response_file"],
                    "text": payload["response"]["text"],
                }
            )
        write_jsonl(self.predictions_path, rows)
        return len(rows)

    # ----------------------------------------------------------------- lock

    def lock_is_held(self) -> bool:
        """True if some process currently holds the writer lock (probe without keeping it).

        Uses a momentary *shared* lock and never creates the lock file. A writer starting at
        the same instant retries briefly (see ``writer_lock``), so probing cannot make it fail.
        """
        if self._lock_fd is not None:
            return True
        import fcntl

        try:
            fd = os.open(self.lock_path, os.O_RDONLY)
        except OSError:
            return False  # never executed: no lock file yet
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    @contextlib.contextmanager
    def writer_lock(self) -> Iterator[None]:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows
            raise RunError("run locking requires fcntl (macOS/Linux)") from None
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        # Retry for ~1 s so a momentary status/evaluate probe is not mistaken for a writer.
        for attempt in range(_LOCK_RETRIES):
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if attempt == _LOCK_RETRIES - 1:
                    os.close(fd)
                    raise RunLockedError(
                        f"run {self.path} is locked by another active writer; "
                        "refusing to start a second"
                    ) from None
                time.sleep(_LOCK_RETRY_INTERVAL_S)
        try:
            os.ftruncate(fd, 0)
            os.write(
                fd, f"pid={os.getpid()} host={socket.gethostname()} at={utc_now_iso()}\n".encode()
            )
            self._lock_fd = fd
            yield
        finally:
            self._lock_fd = None
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _require_lock(self) -> None:
        if self._lock_fd is None:
            raise RunError("write attempted without holding the run lock")
