"""Gemini adapter against a FAKE client (real google-genai ``types``; no network, no key).

These verify request construction and error mapping only. They are not live validation.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

genai_types = pytest.importorskip("google.genai.types")
genai_errors = pytest.importorskip("google.genai.errors")
httpx = pytest.importorskip("httpx")

from video_report.config import GeminiProviderConfig  # noqa: E402
from video_report.providers.base import (  # noqa: E402
    ModelRequest,
    PermanentProviderError,
    ProviderTimeoutError,
    RetryableProviderError,
    VideoInput,
)
from video_report.providers.gemini import GeminiProvider, validate_request  # noqa: E402

KEY = "AIzaFAKEKEY_abcdefghijklmnopqrstuvwxyz012"


class FakeFiles:
    def __init__(self, states: list[str]) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.gets = 0
        self.deleted: list[str] = []
        self._states = states
        self._lock = threading.Lock()

    def upload(self, *, file: str, config: Any) -> Any:
        with self._lock:
            self.uploads.append({"file": file, "config": config})
            n = len(self.uploads)
        return SimpleNamespace(
            name=f"files/{n}", uri=f"https://fake/files/{n}", state=self._states[0]
        )

    def get(self, *, name: str) -> Any:
        self.gets += 1
        state = self._states[min(self.gets, len(self._states) - 1)]
        return SimpleNamespace(name=name, uri=f"https://fake/{name}", state=state)

    def delete(self, *, name: str) -> None:
        self.deleted.append(name)


class FakeModels:
    def __init__(self, behaviour: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.behaviour = behaviour

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        return self.behaviour


def _response(parts: list[Any], usage: Any = None) -> Any:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason=SimpleNamespace(name="STOP"), content=SimpleNamespace(parts=parts)
            )
        ],
        usage_metadata=usage,
        model_version="fake-model-001",
        response_id="resp-1",
        prompt_feedback=None,
    )


def _provider(
    behaviour: Any, states: list[str] | None = None, **opts: Any
) -> tuple[GeminiProvider, Any]:
    client = SimpleNamespace(files=FakeFiles(states or ["ACTIVE"]), models=FakeModels(behaviour))
    p = GeminiProvider(
        "fake-model",
        GeminiProviderConfig(**opts),
        client=client,
        sleep=lambda s: None,
        environ={"GEMINI_API_KEY": KEY},
    )
    return p, client


@pytest.fixture
def video(tmp_path: Path) -> Path:
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"\x00\x00")
    return v


def _request(
    video: Path, call_id: str = "r1::whole", fps: float | None = 2.0, time_range: bool = True
) -> ModelRequest:
    return ModelRequest(
        call_id=call_id,
        report_id="r1",
        field_ids=("f1",),
        system="SYS",
        user="USER",
        videos=(
            VideoInput(
                "cam",
                str(video),
                True,
                "video/mp4",
                12.0 if time_range else None,
                48.5 if time_range else None,
            ),
        ),
        model="fake-model",
        generation={
            "temperature": 0.0,
            "top_p": None,
            "max_output_tokens": 64,
            "seed": 7,
            "response_format": "json",
        },
        fps=fps,
    )


def test_request_construction(video: Path) -> None:
    usage = SimpleNamespace(
        prompt_token_count=1200,
        candidates_token_count=8,
        thoughts_token_count=30,
        cached_content_token_count=None,
        total_token_count=1238,
    )
    text_parts = [
        SimpleNamespace(text="thinking...", thought=True),
        SimpleNamespace(text='{"f1": "A"}', thought=None),
    ]
    p, client = _provider(_response(text_parts, usage))
    resp = p.generate(_request(video), timeout_s=30)

    call = client.models.calls[0]
    assert call["model"] == "fake-model"
    cfg = call["config"]
    assert cfg.system_instruction == "SYS"
    assert cfg.temperature == 0.0 and cfg.max_output_tokens == 64 and cfg.seed == 7
    assert cfg.top_p is None
    assert cfg.response_mime_type == "application/json"
    assert cfg.http_options.timeout == 30_000  # milliseconds
    assert cfg.http_options.retry_options.attempts == 1  # SDK retries disabled
    parts = call["contents"][0].parts
    assert parts[0].file_data.file_uri == "https://fake/files/1"
    assert parts[0].file_data.mime_type == "video/mp4"
    assert parts[0].video_metadata.start_offset == "12.0s"
    assert parts[0].video_metadata.end_offset == "48.5s"
    assert parts[0].video_metadata.fps == 2.0
    assert parts[1].text == "USER"

    assert resp.text == '{"f1": "A"}'  # thought parts excluded
    assert resp.usage is not None
    assert (resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.reasoning_tokens) == (
        1200,
        8,
        30,
    )
    assert resp.finish_reason == "STOP" and resp.model_version == "fake-model-001"


def test_no_video_metadata_when_not_requested(video: Path) -> None:
    p, client = _provider(_response([SimpleNamespace(text="{}", thought=None)]))
    p.generate(_request(video, fps=None, time_range=False), timeout_s=5)
    assert client.models.calls[0]["contents"][0].parts[0].video_metadata is None


def test_upload_once_per_video_and_cleanup(video: Path) -> None:
    p, client = _provider(
        _response([SimpleNamespace(text="{}", thought=None)]),
        states=["PROCESSING", "PROCESSING", "ACTIVE"],
    )
    for i in range(3):
        p.generate(_request(video, call_id=f"c{i}"), timeout_s=5)
    assert len(client.files.uploads) == 1 and client.files.gets == 2
    p.close()
    assert client.files.deleted == ["files/1"]


def test_file_processing_failure_and_timeout(video: Path) -> None:
    p, _ = _provider(_response([]), states=["PROCESSING", "FAILED"])
    with pytest.raises(PermanentProviderError, match="FAILED"):
        p.generate(_request(video), timeout_s=5)
    clock = iter([0.0, 0.0, 1000.0, 1000.0])
    client = SimpleNamespace(files=FakeFiles(["PROCESSING"]), models=FakeModels(_response([])))
    p2 = GeminiProvider(
        "m",
        GeminiProviderConfig(file_active_timeout_s=10),
        client=client,
        sleep=lambda s: None,
        monotonic=lambda: next(clock),
        environ={},
    )
    with pytest.raises(RetryableProviderError, match="not ACTIVE"):
        p2.generate(_request(video), timeout_s=5)


def _api_error(code: int) -> Exception:
    return genai_errors.APIError(
        code, {"error": {"code": code, "message": f"err {KEY}", "status": "X"}}
    )


@pytest.mark.parametrize(
    "exc, cls, kind",
    [
        (_api_error(429), RetryableProviderError, "http_429"),
        (_api_error(503), RetryableProviderError, "http_503"),
        (_api_error(500), RetryableProviderError, "http_500"),
        (_api_error(400), PermanentProviderError, "http_400"),
        (_api_error(403), PermanentProviderError, "http_403"),
        (_api_error(404), PermanentProviderError, "http_404"),
        (httpx.ReadTimeout("read timed out"), ProviderTimeoutError, "timeout"),
        (httpx.ConnectTimeout("connect timed out"), RetryableProviderError, "connect_timeout"),
        (httpx.ConnectError("refused"), RetryableProviderError, "connect_error"),
        (httpx.ReadError("connection reset"), RetryableProviderError, "transport_error"),
        (httpx.RemoteProtocolError("peer closed"), RetryableProviderError, "transport_error"),
        (ValueError("weird"), PermanentProviderError, "unexpected_error"),
    ],
)
def test_error_classification_and_redaction(
    video: Path, exc: Exception, cls: type, kind: str
) -> None:
    p, _ = _provider(exc)
    with pytest.raises(cls) as info:
        p.generate(_request(video), timeout_s=5)
    err = info.value
    assert type(err) is cls and err.kind == kind
    assert KEY not in err.message and KEY not in str(err.to_json()) and KEY not in str(err)
    if cls is ProviderTimeoutError:
        assert err.outcome_unknown is True
    # Only failures that prove the request was never sent count as "not processed".
    never_sent = kind in ("connect_timeout", "connect_error") or kind.startswith("http_")
    assert err.outcome_unknown is (not never_sent)


def test_missing_video_file_is_permanent(tmp_path: Path) -> None:
    p, client = _provider(_response([]))
    with pytest.raises(PermanentProviderError, match="not found"):
        p.generate(_request(tmp_path / "nope.mp4"), timeout_s=5)
    assert client.models.calls == []


def test_unsupported_inputs_rejected(video: Path) -> None:
    remote = ModelRequest(
        "c", "r", ("f",), None, "u", (VideoInput("v", "gs://b/x.mp4", False, "video/mp4"),), "m", {}
    )
    with pytest.raises(ValueError, match="local files only"):
        validate_request(remote)
    odd = ModelRequest(
        "c", "r", ("f",), None, "u", (VideoInput("v", str(video), True, None),), "m", {}
    )
    with pytest.raises(ValueError, match="mime type"):
        validate_request(odd)


def test_missing_credentials() -> None:
    with pytest.raises(PermanentProviderError, match="GEMINI_API_KEY"):
        GeminiProvider("m", GeminiProviderConfig(), environ={})


def test_describe_has_no_secret() -> None:
    p, _ = _provider(_response([]))
    d = p.describe()
    assert KEY not in str(d) and d["sdk_version"] and d["model"] == "fake-model"
