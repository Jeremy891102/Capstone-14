"""Deterministic mock provider for offline runs and tests.

Answers come from a canned ``mock_responses.jsonl`` file (frozen into the run snapshot). The mock
never reads ground truth: its only inputs are that file and the ``ModelRequest``. It ignores
videos, fps, and generation settings, and reports usage as unknown (``None``) rather than
inventing token counts.

mock_responses.jsonl line format::

    {"report_id": "r1", "answers": {"field_a": "B", "field_b": null},
     "raw_by_call": {"r1::field=field_c": "not json"}}

* ``answers``: letter (or any JSON value) per field; ``null`` omits the key from the response.
* ``raw_by_call`` (optional): exact raw text to return for a given call id.

Fault injection for tests: ``provider.mock.fail_plan`` (frozen with the run) scripts per-attempt
failures; the environment variable ``VIDEO_REPORT_MOCK_CRASH_ON_CALL=<call_id>`` makes the
process exit abruptly while that call is in flight. The crash hook is deliberately NOT part of
the frozen config, so a resumed run does not crash again.
"""

from __future__ import annotations

import fnmatch
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from video_report.config import MockProviderConfig
from video_report.io_utils import iter_jsonl
from video_report.providers.base import (
    ModelRequest,
    PermanentProviderError,
    ProviderResponse,
    ProviderTimeoutError,
    RetryableProviderError,
)

MOCK_HARD_EXIT_CODE = 75
CRASH_ENV = "VIDEO_REPORT_MOCK_CRASH_ON_CALL"


def load_mock_responses(path: Path) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for lineno, obj in iter_jsonl(path):
        if not isinstance(obj, dict) or not isinstance(obj.get("report_id"), str):
            raise ValueError(f"{path}:{lineno}: expected an object with a string report_id")
        unknown = set(obj) - {"report_id", "answers", "raw_by_call"}
        if unknown:
            raise ValueError(f"{path}:{lineno}: unknown keys {sorted(unknown)}")
        if obj["report_id"] in table:
            raise ValueError(f"{path}:{lineno}: duplicate report_id {obj['report_id']!r}")
        table[obj["report_id"]] = {
            "answers": dict(obj.get("answers", {})),
            "raw_by_call": dict(obj.get("raw_by_call", {})),
        }
    return table


class MockProvider:
    name = "mock"

    def __init__(
        self,
        model: str,
        responses_path: Path,
        options: MockProviderConfig,
        *,
        sleep: Any = time.sleep,
    ) -> None:
        self.model = model
        self._table = load_mock_responses(responses_path)
        self._options = options
        self._sleep = sleep
        self._attempts: dict[str, int] = {}
        self._lock = threading.Lock()

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "notes": "canned responses; ignores video, fps and generation settings; usage unknown",
        }

    def close(self) -> None:
        return None

    def _next_attempt(self, call_id: str) -> int:
        with self._lock:
            n = self._attempts.get(call_id, 0) + 1
            self._attempts[call_id] = n
            return n

    def _scripted_failure(self, call_id: str, attempt: int) -> str | None:
        for pattern, kinds in self._options.fail_plan.items():
            if fnmatch.fnmatchcase(call_id, pattern) and attempt <= len(kinds):
                return kinds[attempt - 1]
        return None

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ProviderResponse:
        attempt = self._next_attempt(request.call_id)
        if os.environ.get(CRASH_ENV) == request.call_id:
            # Simulates a process crash while a request is in flight.
            os._exit(MOCK_HARD_EXIT_CODE)
        if self._options.delay_s:
            if self._options.delay_s > timeout_s:
                self._sleep(timeout_s)
                raise ProviderTimeoutError("timeout", f"mock delay exceeded {timeout_s}s")
            self._sleep(self._options.delay_s)

        failure = self._scripted_failure(request.call_id, attempt)
        if failure == "retryable":
            raise RetryableProviderError("mock_unavailable", "scripted 503", status_code=503)
        if failure == "timeout":
            raise ProviderTimeoutError("timeout", "scripted timeout")
        if failure == "permanent":
            raise PermanentProviderError("mock_bad_request", "scripted 400", status_code=400)

        entry = self._table.get(request.report_id)
        if entry is None:
            raise PermanentProviderError(
                "mock_no_response", f"no canned response for report {request.report_id!r}"
            )
        if request.call_id in entry["raw_by_call"]:
            text = entry["raw_by_call"][request.call_id]
        else:
            answers = entry["answers"]
            text = json.dumps(
                {f: answers[f] for f in request.field_ids if answers.get(f) is not None}
            )
        return ProviderResponse(
            text=text, usage=None, finish_reason="STOP", model_version=self.model
        )
