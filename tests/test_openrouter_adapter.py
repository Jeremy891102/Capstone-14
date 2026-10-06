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


def _completion(usage=None, content='{"f1": "A"}'):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": usage,
    }


def test_reasoning_is_charged_once_by_evaluator():
    from video_report.config import PricingConfig
    from video_report.evaluation.aggregate import _attempt_cost

    p = _provider(lambda request: httpx.Response(200))
    try:
        result = p._to_response(
            httpx.Response(
                200,
                json=_completion(
                    {
                        "prompt_tokens": 100,
                        "completion_tokens": 30,
                        "completion_tokens_details": {"reasoning_tokens": 20},
                        "total_tokens": 130,
                    }
                ),
            )
        )
        assert result.usage.output_tokens == 10
        assert result.usage.reasoning_tokens == 20
        pricing = PricingConfig(input_per_million=1, output_per_million=2, source="fixture")
        # 100 input + 30 total completion; NOT 100 input + 50 completion.
        assert _attempt_cost(result.usage.to_json(), pricing) == pytest.approx(0.00016)
    finally:
        p.close()


@pytest.mark.parametrize(
    "body",
    [
        [],
        None,
        "Bad gateway",
        {},
        {"choices": []},
        {"choices": "wrong"},
        {"choices": [None]},
        {"choices": [{"message": []}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": {"answer": "A"}}}]},
    ],
)
def test_invalid_success_envelope_is_classified(body):
    p = _provider(lambda request: httpx.Response(200))
    try:
        # json=None means no payload in httpx, exercising invalid JSON as well.
        with pytest.raises(PermanentProviderError) as exc:
            p._to_response(httpx.Response(200, json=body))
        assert exc.value.kind == "invalid_response"
        assert exc.value.outcome_unknown is True
        assert exc.value.retryable is False
    finally:
        p.close()


def test_plain_text_200_is_not_success():
    p = _provider(lambda request: httpx.Response(200))
    try:
        with pytest.raises(PermanentProviderError, match="invalid_response"):
            p._to_response(httpx.Response(200, text="Bad gateway"))
    finally:
        p.close()


@pytest.mark.parametrize("content", ["not JSON", "", None])
def test_model_output_format_is_left_to_evaluator(content):
    p = _provider(lambda request: httpx.Response(200))
    try:
        result = p._to_response(httpx.Response(200, json=_completion(content=content)))
        assert result.text == (content or None)
        assert result.usage is None
    finally:
        p.close()


@pytest.mark.parametrize(
    "usage,expected_output,expected_reasoning",
    [
        ({"completion_tokens": 30}, 30, None),
        ({"completion_tokens": 30, "completion_tokens_details": {"reasoning_tokens": 0}}, 30, 0),
        ({"completion_tokens": 30, "completion_tokens_details": {"reasoning_tokens": 30}}, 0, 30),
        ({"completion_tokens_details": {"reasoning_tokens": 20}}, None, 20),
        ({"prompt_tokens": 100}, None, None),
    ],
)
def test_usage_missing_fields_remain_unknown(usage, expected_output, expected_reasoning):
    p = _provider(lambda request: httpx.Response(200))
    try:
        result = p._to_response(httpx.Response(200, json=_completion(usage)))
        assert result.usage.output_tokens == expected_output
        assert result.usage.reasoning_tokens == expected_reasoning
    finally:
        p.close()


@pytest.mark.parametrize("status", [401, 429, 503])
def test_non_json_http_error_keeps_status_classification(status):
    p = _provider(lambda request: httpx.Response(200))
    try:
        cls = PermanentProviderError if status == 401 else RetryableProviderError
        with pytest.raises(cls) as exc:
            p._to_response(httpx.Response(status, text="upstream failed"))
        assert exc.value.kind == f"http_{status}"
    finally:
        p.close()


@pytest.mark.parametrize(
    "error,cls",
    [
        ({"code": 429, "message": "rate limit"}, RetryableProviderError),
        ({"code": 401, "message": "auth"}, PermanentProviderError),
    ],
)
def test_embedded_error_preserves_classification(error, cls):
    p = _provider(lambda request: httpx.Response(200))
    try:
        with pytest.raises(cls):
            p._to_response(httpx.Response(200, json={"error": error}))
    finally:
        p.close()


@pytest.mark.parametrize(
    "usage",
    [
        [],
        {"completion_tokens": -1},
        {"completion_tokens": True},
        {"completion_tokens": "30"},
        {"completion_tokens_details": []},
        {"prompt_tokens_details": "bad"},
        {"completion_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 20}},
    ],
)
def test_invalid_usage_is_classified(usage):
    p = _provider(lambda request: httpx.Response(200))
    try:
        with pytest.raises(PermanentProviderError) as exc:
            p._to_response(httpx.Response(200, json=_completion(usage)))
        assert exc.value.kind == "invalid_response" and exc.value.outcome_unknown
    finally:
        p.close()


def test_explicit_refusal_is_a_valid_envelope():
    p = _provider(lambda request: httpx.Response(200))
    try:
        result = p._to_response(
            httpx.Response(200, json={"choices": [{"message": {"refusal": "Unable to answer"}}]})
        )
        assert result.text is None
    finally:
        p.close()


def test_malformed_response_is_checkpointed_as_failure_without_retry(
    video, make_bench, make_experiment, tmp_path
):
    import shutil

    from video_report.evaluation.evaluate import evaluate_run
    from video_report.pipeline import create_run, execute_run

    bench = make_bench({"r1": ["f1"]})
    (bench / "videos").mkdir()
    shutil.copyfile(video, bench / "videos/r1.mp4")
    reports = bench / "reports.jsonl"
    row = json.loads(reports.read_text())
    row["input"]["videos"][0]["time_range"] = {
        "start": 1,
        "end": 3,
        "unit": "seconds",
        "reference": "video_start",
    }
    reports.write_text(json.dumps(row) + "\n")
    config = make_experiment(
        bench, provider__name="openrouter", provider__model="google/x", execution__max_attempts=3
    )
    run = create_run(config, tmp_path / "runs", environ={"OPENROUTER_API_KEY": "fake-key"})
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, text="Bad gateway")

    result = execute_run(run, mode="create", provider_factory=lambda cfg, run: _provider(handler))
    assert result.state == "completed_with_failures"
    assert len(seen) == 1  # despite a retry budget of 3
    call = run.read_all_calls()[0]
    assert call["status"] == "failed"
    assert call["attempts"][0]["error"]["kind"] == "invalid_response"
    assert call["attempts"][0]["error"]["outcome_unknown"] is True
    assert not list(run.responses_dir.rglob("attempt-*.json"))
    _, metrics = evaluate_run(run.path, bench / "ground_truth.jsonl")
    assert metrics["execution"]["execution_completion"]["numerator"] == 0
    assert metrics["fields"]["outcomes"]["not_executed"] == 1
    assert metrics["usage"]["attempts"]["possibly_billed"] == 1
