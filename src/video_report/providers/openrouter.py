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
            data = {}
        err = data.get("error") if isinstance(data, dict) else None
        if status != 200 or (err and not data.get("choices")):
            # OpenRouter may also report upstream failures as HTTP 200 with an "error" object.
            code = status if status != 200 else (err or {}).get("code", 502)
            msg = redact(f"{status}: {err or resp.text[:300]}", (self._api_key,))
            cls = RetryableProviderError if code in _RETRYABLE_STATUS else PermanentProviderError
            return_err = cls(f"http_{code}", msg, status_code=status)
            raise return_err
        choice = (data.get("choices") or [{}])[0]
        u = data.get("usage") or {}
        usage = Usage(
            input_tokens=u.get("prompt_tokens"),
            output_tokens=u.get("completion_tokens"),
            reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            cached_input_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            total_tokens=u.get("total_tokens"),
        )
        return ProviderResponse(
            text=(choice.get("message") or {}).get("content") or None,
            usage=usage if u else None,
            finish_reason=choice.get("finish_reason"),
            model_version=data.get("model"),
            response_id=data.get("id"),
        )
