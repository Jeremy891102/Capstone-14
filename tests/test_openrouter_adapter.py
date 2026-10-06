"""OpenRouter adapter against a FAKE httpx transport (real ffmpeg clip, no network, no key)."""

from __future__ import annotations

import json
import subprocess

import pytest

httpx = pytest.importorskip("httpx")
imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")

from video_report.config import OpenRouterProviderConfig  # noqa: E402
from video_report.providers.base import (  # noqa: E402
    ModelRequest,
    PermanentProviderError,
    RetryableProviderError,
    VideoInput,
)
from video_report.providers.openrouter import OpenRouterProvider  # noqa: E402


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "src.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=4:size=320x240:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(p),
        ],
        check=True,
    )
    return p


def _req(video, start=1.0, end=3.0) -> ModelRequest:
    v = VideoInput("head_cam", str(video), True, "video/mp4", start, end)
    return ModelRequest(
        "c1",
        "r1",
        ("f1",),
        "sys",
        "question?",
        (v,),
        "google/x",
        {"temperature": 0.0, "max_output_tokens": 50, "response_format": "json"},
    )


def _provider(handler) -> OpenRouterProvider:
    client = httpx.Client(base_url="https://x/api/v1", transport=httpx.MockTransport(handler))
    return OpenRouterProvider("google/x", OpenRouterProviderConfig(), client=client)


def test_request_shape_and_response(video):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "model": "google/x",
                "choices": [{"message": {"content": '{"f1": "A"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            },
        )

    p = _provider(handler)
    r = p.generate(_req(video), timeout_s=30)
    p.close()
    assert r.text == '{"f1": "A"}' and r.usage.input_tokens == 10
    assert seen["response_format"] == {"type": "json_object"} and seen["max_tokens"] == 50
    parts = seen["messages"][1]["content"]
    assert parts[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert parts[1] == {"type": "text", "text": "question?"}


@pytest.mark.parametrize(
    "status,cls", [(429, RetryableProviderError), (401, PermanentProviderError)]
)
def test_http_errors(video, status, cls):
    p = _provider(lambda request: httpx.Response(status, json={"error": {"message": "no"}}))
    with pytest.raises(cls):
        p.generate(_req(video), timeout_s=30)
    p.close()


def test_requires_time_range(video):
    p = _provider(lambda request: httpx.Response(200))
    with pytest.raises(PermanentProviderError):
        p.generate(_req(video, None, None), timeout_s=30)
    p.close()
