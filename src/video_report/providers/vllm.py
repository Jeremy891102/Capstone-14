"""vLLM adapter (OpenAI-compatible ``/chat/completions``) for Qwen3-VL: ``video_url`` input.

Each clip is cut with ffmpeg at a fixed fps (no audio). Qwen3-VL turns every 2 frames into one
group of ``(h/32) * (w/32)`` tokens, so a clip that would not fit ``video_token_budget`` at
``max_height`` is downscaled instead of subsampled in time. The chosen heights are returned in
the response metadata. The model sees a clip that starts at 0 s, as with the other providers.

The server must keep every frame of the pre-cut clip: start it with
``--media-io-kwargs '{"video": {"fps": <frame_fps>, "num_frames": -1}}'``.
"""

from __future__ import annotations

import base64
import importlib.metadata
import math
import os
from dataclasses import replace
from typing import Any

from video_report.config import VLLMProviderConfig
from video_report.providers.base import (
    ModelRequest,
    PermanentProviderError,
    ProviderResponse,
    Usage,
)
from video_report.providers.http_api import (
    JSONHTTP,
    invalid_response,
    object_or_empty,
    optional_string,
    tokens,
)
from video_report.providers.media import LocalMedia, validate_local_ranges

ASPECT = 16 / 9  # ponytail: assumes 16:9 sources (CC4D GoPro); read the real aspect if that changes


def video_tokens(duration_s: float, fps: float, height: int) -> int:
    """Estimated Qwen3-VL tokens: 2 frames per group, 32 px per token (rounded up)."""
    width = math.ceil(height * ASPECT / 32) * 32
    return math.ceil(duration_s * fps / 2) * (math.ceil(height / 32) * (width // 32))


def choose_height(request: ModelRequest, opts: VLLMProviderConfig) -> int:
    fps = request.fps or opts.frame_fps
    seconds = sum(v.end_s - v.start_s for v in request.videos)  # type: ignore[operator]
    for height in range(opts.max_height, opts.min_height - 1, -32):
        if video_tokens(seconds, fps, height) <= opts.video_token_budget:
            return height
    raise ValueError(
        f"{seconds:.0f}s at {fps} fps exceeds the video token budget even at "
        f"{opts.min_height}p; lower fps or raise video_token_budget explicitly"
    )


def validate_request(request: ModelRequest, options: VLLMProviderConfig | None = None) -> None:
    validate_local_ranges(request)
    choose_height(request, options or VLLMProviderConfig())


class VLLMProvider:
    name = "vllm"

    def __init__(
        self,
        model: str,
        options: VLLMProviderConfig,
        *,
        client: Any = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.options = options
        env = dict(os.environ if environ is None else environ)
        env.setdefault(options.api_key_env, "EMPTY")  # vLLM ignores the key unless --api-key
        self.http = JSONHTTP(options.api_key_env, client=client, environ=env)
        self.media = LocalMedia(options.max_height, options.media_timeout_s)

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "base_url": self.options.base_url,
            "adapter_version": 2,
            "httpx_version": importlib.metadata.version("httpx"),
            "imageio_ffmpeg_version": importlib.metadata.version("imageio-ffmpeg"),
            "input_mode": "video_url_no_audio",
            "options": self.options.model_dump(),
        }

    def close(self) -> None:
        try:
            self.media.close()
        finally:
            self.http.close()

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ProviderResponse:
        try:
            validate_request(request, self.options)
        except ValueError as exc:
            raise PermanentProviderError("unsupported_input", str(exc)) from None
        fps = request.fps or self.options.frame_fps
        height = choose_height(request, self.options)
        content: list[dict[str, Any]] = []
        total_bytes = 0
        for video in request.videos:
            raw = self.media.clip(video, fps, height).read_bytes()
            total_bytes += len(raw)
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(raw).decode()},
                }
            )
        content.append({"type": "text", "text": request.user})
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": content})
        body: dict[str, Any] = {"model": request.model, "messages": messages}
        for src, dst in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("max_output_tokens", "max_tokens"),
            ("seed", "seed"),
        ):
            if request.generation.get(src) is not None:
                body[dst] = request.generation[src]
        if request.generation.get("response_format") == "json":
            body["response_format"] = {"type": "json_object"}
        data = self.http.post(
            self.options.base_url.rstrip("/") + "/chat/completions",
            {"Authorization": f"Bearer {self.http.key}"},
            body,
            timeout_s,
            self.options.max_payload_bytes,
        )
        seconds = sum(v.end_s - v.start_s for v in request.videos)  # type: ignore[operator]
        return replace(
            self._to_response(data),
            metadata={
                "input_mode": "video_url_no_audio",
                "fps": fps,
                "height": height,
                "clip_seconds": seconds,
                "est_video_tokens": video_tokens(seconds, fps, height),
                "clip_bytes": total_bytes,
            },
        )

    @staticmethod
    def _to_response(data: dict[str, Any]) -> ProviderResponse:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise invalid_response("choices must be a nonempty array of objects")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise invalid_response("choice.message must be an object")
        text = optional_string(message.get("content"))
        raw = object_or_empty(data.get("usage"))
        usage = None
        if raw:
            usage = Usage(
                input_tokens=tokens(raw, "prompt_tokens"),
                output_tokens=tokens(raw, "completion_tokens"),
                total_tokens=tokens(raw, "total_tokens"),
            )
        return ProviderResponse(
            text=text or None,
            usage=usage,
            finish_reason=optional_string(choice.get("finish_reason")),
            model_version=optional_string(data.get("model")),
            response_id=optional_string(data.get("id")),
        )
