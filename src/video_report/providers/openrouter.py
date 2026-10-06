"""OpenRouter adapter (OpenAI-compatible ``/chat/completions`` with a ``video_url`` part).

OpenRouter has no file upload and no clip offsets, so every video must be a local file with a
report time range: the range is cut with ffmpeg (bundled via ``imageio-ffmpeg``), downscaled,
stripped of audio, and sent inline as a base64 data URL. The model therefore sees a clip that
starts at 0 s, not the original timeline.
"""

from __future__ import annotations

import base64
import contextlib
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from video_report.config import OpenRouterProviderConfig
from video_report.providers.base import (
    ModelRequest,
    PermanentProviderError,
    ProviderError,
    ProviderResponse,
    ProviderTimeoutError,
    RetryableProviderError,
    Usage,
    VideoInput,
    redact,
)

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def validate_request(request: ModelRequest) -> None:
    """Reject inputs this adapter does not implement (called at run creation)."""
    for v in request.videos:
        if not v.is_local:
            raise ValueError(f"{request.call_id}: video {v.video_id!r} must be a local file")
        if v.start_s is None or v.end_s is None:
            raise ValueError(
                f"{request.call_id}: video {v.video_id!r} needs a time_range "
                "(openrouter sends clips inline; set video.use_time_range: true)"
            )


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())


class OpenRouterProvider:
    name = "openrouter"

    def __init__(
        self,
        model: str,
        options: OpenRouterProviderConfig,
        *,
        client: Any = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        import os

        self.model = model
        self._options = options
        env = os.environ if environ is None else environ
        self._api_key = env.get(options.api_key_env, "")
        if client is None:
            if not self._api_key:
                raise PermanentProviderError(
                    "missing_credentials", f"environment variable {options.api_key_env} is not set"
                )
            import httpx

            client = httpx.Client(
                base_url=options.base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        self._client = client
        self._tmp = Path(tempfile.mkdtemp(prefix="video_report_clips_"))
        self._clips: dict[tuple[Any, ...], str] = {}
        self._lock = threading.Lock()

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "base_url": self._options.base_url,
            "clip_height": self._options.clip_height,
            "clip_fps": self._options.clip_fps,
            "api_key_env": self._options.api_key_env,
        }

    def close(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)
        with contextlib.suppress(Exception):
            self._client.close()

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ProviderResponse:
        try:
            validate_request(request)
        except ValueError as exc:
            raise PermanentProviderError("unsupported_input", str(exc)) from None
        try:
            content: list[dict[str, Any]] = [
                self._video_part(v, request.fps) for v in request.videos
            ]
            content.append({"type": "text", "text": request.user})
            messages: list[dict[str, Any]] = []
            if request.system:
                messages.append({"role": "system", "content": request.system})
            messages.append({"role": "user", "content": content})
            body: dict[str, Any] = {"model": request.model, "messages": messages}
            g = request.generation
            for src, dst in (
                ("temperature", "temperature"),
                ("top_p", "top_p"),
                ("max_output_tokens", "max_tokens"),
                ("seed", "seed"),
            ):
                if g.get(src) is not None:
                    body[dst] = g[src]
            if g.get("response_format") == "json":
                body["response_format"] = {"type": "json_object"}
            resp = self._client.post("/chat/completions", json=body, timeout=timeout_s)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._classify(exc) from None
        return self._to_response(resp)

    # ----------------------------------------------------------------- helpers

    def _video_part(self, video: VideoInput, fps: float | None) -> dict[str, Any]:
        path = self._clip(video, fps or self._options.clip_fps)
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}

    def _clip(self, video: VideoInput, fps: float) -> str:
        src = Path(video.location)
        if not src.exists():
            raise PermanentProviderError(
                "video_not_found", f"video {video.video_id!r} not found at {src}"
            )
        assert video.start_s is not None and video.end_s is not None
        key = (str(src), src.stat().st_mtime_ns, video.start_s, video.end_s, fps)
        with self._lock:  # one ffmpeg at a time; clips are tiny, so no per-key locking
            if key in self._clips:
                return self._clips[key]
            out = self._tmp / f"clip{len(self._clips)}.mp4"
            cmd = [
                _ffmpeg(), "-y", "-loglevel", "error",
                "-ss", str(video.start_s), "-t", str(video.end_s - video.start_s),
                "-i", str(src),
                "-vf", f"scale=-2:{self._options.clip_height}", "-r", str(fps),
                "-an", "-c:v", "libx264", "-crf", "30", "-preset", "veryfast", str(out),
            ]  # fmt: skip
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise PermanentProviderError("clip_failed", f"ffmpeg: {r.stderr[-300:]}")
            self._clips[key] = str(out)
            return str(out)

    def _classify(self, exc: Exception) -> ProviderError:
        msg = redact(f"{type(exc).__name__}: {exc}", (self._api_key,))
        import httpx

        if isinstance(exc, httpx.ConnectTimeout):
            return RetryableProviderError("connect_timeout", msg)
        if isinstance(exc, httpx.ConnectError):
            return RetryableProviderError("connect_error", msg)
        if isinstance(exc, httpx.TimeoutException):
            return ProviderTimeoutError("timeout", msg)
        if isinstance(exc, httpx.TransportError):
            return RetryableProviderError("transport_error", msg, outcome_unknown=True)
        return PermanentProviderError("unexpected_error", msg, outcome_unknown=True)

    def _to_response(self, resp: Any) -> ProviderResponse:
        status = resp.status_code
        try:
            data = resp.json()
        except ValueError:
            data = None

        # HTTP failures keep their status classification, even with non-JSON bodies.
        # A 200 response must pass envelope validation before it can become a success.
        err = data.get("error") if isinstance(data, dict) else None
        if status != 200 or err is not None:
            code = status
            if status == 200:
                if not isinstance(err, dict):
                    raise self._invalid_response("error must be an object")
                code = err.get("code", 502)
            if type(code) is not int:
                raise self._invalid_response("error code must be an integer")
            # Do not persist arbitrary gateway bodies, which can echo input or credentials.
            cls = RetryableProviderError if code in _RETRYABLE_STATUS else PermanentProviderError
            raise cls(
                f"http_{code}",
                f"OpenRouter returned error {code} (HTTP {status})",
                status_code=status,
                outcome_unknown=status == 200,
            )

        try:
            if not isinstance(data, dict):
                raise ValueError("response must be a JSON object")
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("choices must be a nonempty array of objects")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict):
                raise ValueError("choice.message must be an object")
            if "content" not in message and not isinstance(message.get("refusal"), str):
                raise ValueError("message must contain content or an explicit refusal")
            text = message.get("content")
            if text is not None and not isinstance(text, str):
                raise ValueError("message.content must be a string or null")
            # This validates the API envelope, NOT the model's answer format. Empty, null,
            # or non-JSON model text still reaches the existing offline parser/scorer.
            usage = self._usage(data.get("usage"))
            for value in (choice.get("finish_reason"), data.get("model"), data.get("id")):
                if value is not None and not isinstance(value, str):
                    raise ValueError("response identifiers must be strings or null")
        except ValueError as exc:
            raise self._invalid_response(str(exc)) from None
        return ProviderResponse(
            text=text or None,
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            model_version=data.get("model"),
            response_id=data.get("id"),
        )

    @staticmethod
    def _invalid_response(detail: str) -> PermanentProviderError:
        # The remote call may have succeeded and been billed. Do not automatically resend
        # a malformed success response; make the failed attempt visible in checkpoints.
        return PermanentProviderError(
            "invalid_response", detail, status_code=200, outcome_unknown=True
        )

    @staticmethod
    def _usage(raw: Any) -> Usage | None:
        if raw is None or raw == {}:
            return None
        if not isinstance(raw, dict):
            raise ValueError("usage must be an object or null")

        def tokens(obj: dict[str, Any], key: str) -> int | None:
            value = obj.get(key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"usage.{key} must be a nonnegative integer or null")
            return value

        details = raw.get("completion_tokens_details")
        prompt_details = raw.get("prompt_tokens_details")
        if details is None:
            details = {}
        if prompt_details is None:
            prompt_details = {}
        if not isinstance(details, dict) or not isinstance(prompt_details, dict):
            raise ValueError("usage token details must be objects or null")
        completion = tokens(raw, "completion_tokens")
        reasoning = tokens(details, "reasoning_tokens")
        output = completion
        if completion is not None and reasoning is not None:
            if reasoning > completion:
                raise ValueError("reasoning_tokens exceeds completion_tokens")
            # OpenRouter includes reasoning in completion_tokens. The evaluator adds the
            # two fields, so store only non-reasoning output here to charge the total once.
            output = completion - reasoning
        # If the reasoning breakdown is absent, preserve the reported completion total;
        # do not fabricate a zero reasoning count. Missing completion remains unknown.
        return Usage(
            input_tokens=tokens(raw, "prompt_tokens"),
            output_tokens=output,
            reasoning_tokens=reasoning,
            cached_input_tokens=tokens(prompt_details, "cached_tokens"),
            total_tokens=tokens(raw, "total_tokens"),
        )
