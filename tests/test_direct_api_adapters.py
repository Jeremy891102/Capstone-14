"""Direct providers: synthetic media + fake HTTP only; no keys or paid calls."""

from __future__ import annotations

import base64
import json
import subprocess
from dataclasses import replace

import pytest

httpx = pytest.importorskip("httpx")
imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")

from video_report.config import (  # noqa: E402
    GeminiProviderConfig,
    OpenAIProviderConfig,
    ProviderConfig,
)
from video_report.providers import build_provider, preflight  # noqa: E402
from video_report.providers.base import (  # noqa: E402
    ModelRequest,
    PermanentProviderError,
    ProviderTimeoutError,
    RetryableProviderError,
    VideoInput,
)
from video_report.providers.http_api import JSONHTTP  # noqa: E402
from video_report.providers.media import LocalMedia  # noqa: E402
from video_report.providers.openai import OpenAIProvider, validate_request  # noqa: E402
from video_report.providers.vertex_gemini import BASE, VertexGeminiProvider  # noqa: E402


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "video.mp4"
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


def req(video):
    return ModelRequest(
        "r::whole",
        "r",
        ("f",),
        "Return JSON",
        "Question in JSON",
        (VideoInput("cam", str(video), True, "video/mp4", 1, 3),),
        "test-model",
        {"max_output_tokens": 64, "response_format": "json"},
        fps=2,
    )


def oa_response(**extra):
    return {
        "status": "completed",
        "id": "resp-fake",
        "model": "test-model",
        "output": [
            {"type": "reasoning"},
            {"type": "message", "content": [{"type": "output_text", "text": '{"f":"A"}'}]},
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 20},
            "total_tokens": 130,
        },
        **extra,
    }


def vg_response(**extra):
    return {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"text": "hidden", "thought": True}, {"text": '{"f":"A"}'}]},
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 10,
            "thoughtsTokenCount": 20,
            "totalTokenCount": 130,
        },
        **extra,
    }


def test_openai_payload_frames_usage_and_cleanup(video):
    seen = []

    def handler(r):
        seen.append(r)
        return httpx.Response(200, json=oa_response())

    p = OpenAIProvider(
        "test-model",
        OpenAIProviderConfig(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        environ={"OPENAI_API_KEY": "fake-openai-key"},
    )
    root = p.media.root
    try:
        result = p.generate(req(video), timeout_s=4)
        body = json.loads(seen[0].content)
        assert str(seen[0].url) == "https://api.openai.com/v1/responses"
        assert seen[0].headers["authorization"] == "Bearer fake-openai-key"
        assert body["store"] is False and body["text"]["format"]["type"] == "json_object"
        assert body["max_output_tokens"] == 64 and body["instructions"] == "Return JSON"
        content = body["input"][0]["content"]
        images = [x for x in content if x["type"] == "input_image"]
        assert len(images) == 4
        assert base64.b64decode(images[0]["image_url"].split(",")[1]).startswith(b"\xff\xd8")
        labels = [x["text"] for x in content if x["type"] == "input_text"][1:]
        assert "0.000 seconds" in labels[0] and "1.500 seconds" in labels[-1]
        assert result.text == '{"f":"A"}'
        assert result.usage.output_tokens == 10 and result.usage.reasoning_tokens == 20
        assert "fake-openai-key" not in json.dumps(p.describe())
    finally:
        p.close()
    assert not root.exists()


def test_vertex_matches_dennis_endpoint_and_preserves_model(video):
    seen = []

    def handler(r):
        seen.append(r)
        return httpx.Response(200, json=vg_response())

    options = GeminiProviderConfig(backend="vertex_express")
    p = VertexGeminiProvider(
        "gemini-3.5-flash-lite",
        options,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        environ={"GEMINI_API_KEY": "fake-google-key"},
    )
    try:
        result = p.generate(replace(req(video), model="gemini-3.5-flash-lite"), timeout_s=4)
        assert str(seen[0].url) == BASE + "gemini-3.5-flash-lite:generateContent"
        assert seen[0].headers["x-goog-api-key"] == "fake-google-key"
        body = json.loads(seen[0].content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        parts = body["contents"][0]["parts"]
        assert parts[1]["inlineData"]["mimeType"] == "video/mp4"
        assert base64.b64decode(parts[1]["inlineData"]["data"])[4:8] == b"ftyp"
        assert result.text == '{"f":"A"}'
        assert result.usage.output_tokens == 10 and result.usage.reasoning_tokens == 20
        assert p.describe()["backend"] == "vertex_express"
    finally:
        p.close()


@pytest.mark.parametrize(
    "status,cls",
    [(401, PermanentProviderError), (429, RetryableProviderError), (503, RetryableProviderError)],
)
def test_http_errors_do_not_retry_or_echo_secrets(status, cls):
    seen = []

    def handler(r):
        seen.append(r)
        return httpx.Response(status, text="secret-body")

    api = JSONHTTP("TEST_KEY", client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        with pytest.raises(cls) as exc:
            api.post("https://fake.test", {}, {}, 1, 1000)
        assert len(seen) == 1 and "secret-body" not in str(exc.value)
    finally:
        api.close()


@pytest.mark.parametrize("body", [[], "bad gateway", None, {"error": {"code": 503}}])
def test_bad_http_200_envelopes(body):
    api = JSONHTTP(
        "TEST_KEY",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
        ),
    )
    try:
        with pytest.raises(PermanentProviderError) as exc:
            api.post("https://fake.test", {}, {}, 1, 1000)
        assert exc.value.kind == "invalid_response" and exc.value.outcome_unknown
    finally:
        api.close()


def test_guards_fail_before_http(video):
    request = req(video)
    with pytest.raises(ValueError, match="seed"):
        validate_request(replace(request, generation={"seed": 7}))
    with pytest.raises(ValueError, match="max_frames"):
        validate_request(request, OpenAIProviderConfig(max_frames=1))
    with pytest.raises(ValueError, match="time_range"):
        validate_request(replace(request, videos=(replace(request.videos[0], start_s=None),)))
    media = LocalMedia(480, 5)
    try:
        with pytest.raises(PermanentProviderError, match="frame_limit"):
            media.frames(request.videos[0], 2, 1)
    finally:
        media.close()


@pytest.mark.parametrize(
    "name,extra", [("openai", {}), ("gemini", {"gemini": {"backend": "vertex_express"}})]
)
def test_preflight_and_factory(name, extra):
    cfg = ProviderConfig(name=name, model="test-model", **extra)
    with pytest.raises(ValueError, match="API_KEY"):
        preflight(cfg, {})
    env = {"OPENAI_API_KEY": "fake", "GEMINI_API_KEY": "fake"}
    preflight(cfg, env)


def test_factory_routes_vertex_without_google_sdk(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    cfg = ProviderConfig(name="gemini", model="test-model", gemini={"backend": "vertex_express"})
    p = build_provider(cfg, None)
    assert isinstance(p, VertexGeminiProvider)
    p.close()


@pytest.mark.parametrize(
    "provider,data",
    [
        (OpenAIProvider, {"status": "completed", "output": [None]}),
        (OpenAIProvider, {"status": "completed", "output": [], "usage": {"output_tokens": -1}}),
        (VertexGeminiProvider, {"candidates": "bad"}),
        (VertexGeminiProvider, {"candidates": [{"content": {"parts": [42]}}]}),
    ],
)
def test_provider_envelope_guards(provider, data):
    with pytest.raises(PermanentProviderError) as exc:
        provider._to_response(data)
    assert exc.value.kind == "invalid_response"


def test_null_model_output_and_block_reason():
    data = oa_response(
        output=[{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]
    )
    assert OpenAIProvider._to_response(data).text is None
    assert (
        VertexGeminiProvider._to_response({"promptFeedback": {"blockReason": "SAFETY"}}).text
        is None
    )


def test_timeout_unknown_and_payload_guard():
    def handler(r):
        raise httpx.ReadTimeout("secret")

    api = JSONHTTP("TEST_KEY", client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        with pytest.raises(ProviderTimeoutError):
            api.post("https://fake.test", {}, {}, 1, 1000)
        with pytest.raises(PermanentProviderError, match="payload_limit"):
            api.post("https://fake.test", {}, {"huge": "value"}, 1, 1)
    finally:
        api.close()


@pytest.mark.parametrize("kind", ["openai", "vertex"])
def test_direct_provider_full_pipeline_offline(kind, video, make_bench, make_experiment, tmp_path):
    import shutil

    from video_report.config import PricingConfig
    from video_report.evaluation.evaluate import evaluate_run
    from video_report.pipeline import create_run, execute_run

    bench = make_bench({"r": ["f"]})
    (bench / "videos").mkdir()
    shutil.copyfile(video, bench / "videos/r.mp4")
    reports = bench / "reports.jsonl"
    row = json.loads(reports.read_text())
    row["input"]["videos"][0]["time_range"] = {
        "start": 1,
        "end": 3,
        "unit": "seconds",
        "reference": "video_start",
    }
    reports.write_text(json.dumps(row) + "\n")
    options = {
        "provider__name": "openai" if kind == "openai" else "gemini",
        "provider__model": "test-model",
    }
    if kind == "vertex":
        options["provider__gemini"] = {"backend": "vertex_express"}
    config = make_experiment(bench, **options)
    run = create_run(
        config, tmp_path / "runs", environ={"OPENAI_API_KEY": "fake", "GEMINI_API_KEY": "fake"}
    )
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=oa_response() if kind == "openai" else vg_response())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = (
        OpenAIProvider("test-model", OpenAIProviderConfig(), client=client)
        if kind == "openai"
        else VertexGeminiProvider(
            "test-model", GeminiProviderConfig(backend="vertex_express"), client=client
        )
    )
    assert (
        execute_run(run, mode="create", provider_factory=lambda cfg, run: provider).state
        == "completed"
    )
    assert len(seen) == 1
    _, metrics = evaluate_run(
        run.path,
        bench / "ground_truth.jsonl",
        pricing=PricingConfig(input_per_million=1, output_per_million=2, source="offline test"),
    )
    assert metrics["fields"]["field_accuracy"]["value"] == 1
    assert metrics["usage"]["cost"] == pytest.approx(0.00016)
    assert "API_KEY" not in json.dumps(next(iter(run.load_requests().values())).to_json())


def test_missing_real_credentials():
    for ctor, options in [
        (OpenAIProvider, OpenAIProviderConfig()),
        (VertexGeminiProvider, GeminiProviderConfig(backend="vertex_express")),
    ]:
        with pytest.raises(PermanentProviderError, match="missing_credentials"):
            ctor("test-model", options, environ={})


@pytest.mark.parametrize(
    "data",
    [
        {"status": [], "output": []},
        {"status": "completed", "output": [{"type": "message", "content": [42]}]},
        {
            "status": "completed",
            "output": [],
            "usage": {"output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 2}},
        },
    ],
)
def test_openai_malformed_response_fields(data):
    with pytest.raises(PermanentProviderError, match="invalid_response"):
        OpenAIProvider._to_response(data)
