from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from video_report.providers.base import (
    ModelRequest,
    PermanentProviderError,
    ProviderResponse,
    ProviderTimeoutError,
    RetryableProviderError,
    Usage,
)

REPO = Path(__file__).resolve().parents[1]
MOCK_BENCH = REPO / "benchmarks" / "mock_mcq_v1"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def make_field(fid: str, n_choices: int = 4, **kw: Any) -> dict[str, Any]:
    return {
        "id": fid,
        "question": f"Question about {fid}?",
        "answer_type": "single_choice",
        "choices": [f"{fid} option {i}" for i in range(n_choices)],
        **kw,
    }


def make_report(rid: str, field_ids: list[str], **kw: Any) -> dict[str, Any]:
    return {
        "id": rid,
        "input": {
            "videos": [{"id": "cam", "uri": f"videos/{rid}.mp4", "mime_type": "video/mp4"}],
            "context": {"task_name": f"task for {rid}", "secret_hint": "DO_NOT_SEND"},
        },
        "fields": [make_field(f) for f in field_ids],
        "source": {"dataset": "synthetic", "annotation_refs": [f"ann://{rid}"]},
        "metadata": {"note": "METADATA_SENTINEL"},
        **kw,
    }


@pytest.fixture
def mock_bench(tmp_path: Path) -> Path:
    """A private copy of the committed synthetic benchmark."""
    dst = tmp_path / "bench"
    shutil.copytree(MOCK_BENCH, dst)
    return dst


@pytest.fixture
def make_bench(tmp_path: Path) -> Callable[..., Path]:
    """Build a benchmark with N reports x given fields, plus mock answers and ground truth."""

    def _make(
        reports: dict[str, list[str]],
        targets: dict[tuple[str, str], str] | None = None,
        answers: dict[str, dict[str, Any]] | None = None,
        raw_by_call: dict[str, dict[str, str]] | None = None,
        name: str = "bench",
    ) -> Path:
        root = tmp_path / name
        root.mkdir()
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "video_report.v1",
                    "benchmark_id": name,
                    "mock_responses_file": "mock_responses.jsonl",
                }
            )
        )
        write_jsonl(root / "reports.jsonl", [make_report(r, f) for r, f in reports.items()])
        targets = targets or {(r, f): "A" for r, fs in reports.items() for f in fs}
        write_jsonl(
            root / "ground_truth.jsonl",
            [
                {"report_id": r, "field_id": f, "target": t, "evidence": {"x": "EVIDENCE_SENTINEL"}}
                for (r, f), t in targets.items()
            ],
        )
        answers = answers or {r: {f: "A" for f in fs} for r, fs in reports.items()}
        write_jsonl(
            root / "mock_responses.jsonl",
            [
                {"report_id": r, "answers": a, "raw_by_call": (raw_by_call or {}).get(r, {})}
                for r, a in answers.items()
            ],
        )
        return root

    return _make


@pytest.fixture
def make_experiment(tmp_path: Path) -> Callable[..., Path]:
    """Write an experiment directory (config + prompts); returns the config path."""

    def _make(
        bench: Path,
        name: str = "exp_test",
        method: str = "whole_report",
        user_prompt: str | None = None,
        **overrides: Any,
    ) -> Path:
        d = tmp_path / "experiments" / name
        (d / "prompts").mkdir(parents=True)
        (d / "prompts" / "system.txt").write_text("System prompt for $num_fields fields.\n")
        (d / "prompts" / "user.txt").write_text(
            user_prompt or "Context:\n$context\n\n$questions\n\nAnswer as JSON:\n$answer_format\n"
        )
        cfg: dict[str, Any] = {
            "name": name,
            "data": {"benchmark": str(bench)},
            "method": {"name": method},
            "prompt": {
                "system": "prompts/system.txt",
                "user": "prompts/user.txt",
                "context_keys": ["task_name"],
            },
            "provider": {"name": "mock", "model": "mock-test"},
            "execution": {"concurrency": 2, "backoff_initial_s": 0.0, "backoff_jitter": 0.0},
        }
        for key, value in overrides.items():
            section, _, sub = key.partition("__")
            if sub:
                cfg.setdefault(section, {})[sub] = value
            else:
                cfg[section] = value
        path = d / "config.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    return _make


class ScriptedProvider:
    """Thread-safe test provider: per-call scripts of failures/responses, call log, delays."""

    name = "scripted"

    def __init__(
        self,
        script: dict[str, list[Any]] | None = None,
        default: Callable[[ModelRequest], str] | None = None,
        delays: dict[str, float] | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.model = "scripted-model"
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.default = default or (lambda r: json.dumps({f: "A" for f in r.field_ids}))
        self.delays = delays or {}
        self.usage = usage
        self.calls: list[str] = []
        self.completed: list[str] = []
        self.requests: list[ModelRequest] = []
        self._lock = threading.Lock()

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model}

    def close(self) -> None:
        pass

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ProviderResponse:
        with self._lock:
            self.calls.append(request.call_id)
            self.requests.append(request)
            steps = self.script.get(request.call_id)
            step = steps.pop(0) if steps else None
        delay = self.delays.get(request.call_id)
        if delay:
            threading.Event().wait(delay)
        if isinstance(step, BaseException):
            raise step
        if step == "retryable":
            raise RetryableProviderError("http_503", "unavailable", status_code=503)
        if step == "timeout":
            raise ProviderTimeoutError("timeout", "read timeout")
        if step == "permanent":
            raise PermanentProviderError("http_400", "bad request", status_code=400)
        text = step if isinstance(step, str) else self.default(request)
        with self._lock:
            self.completed.append(request.call_id)
        return ProviderResponse(text=text, usage=self.usage, finish_reason="STOP")


class SimulatedCrash(BaseException):
    """Escapes the runner like a hard crash would (not an Exception subclass)."""


@pytest.fixture
def no_sleep() -> Callable[[float], None]:
    delays: list[float] = []

    def _sleep(s: float) -> None:
        delays.append(s)

    _sleep.delays = delays  # type: ignore[attr-defined]
    return _sleep
