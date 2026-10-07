"""OpenAI Responses adapter: timestamped JPEG frames, no native video/audio input."""

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import replace
from typing import Any

from video_report.config import OpenAIProviderConfig
from video_report.providers.base import ModelRequest, ProviderResponse, Usage
from video_report.providers.http_api import (
    JSONHTTP,
    invalid_response,
    object_or_empty,
    optional_string,
    tokens,
)
from video_report.providers.media import LocalMedia, encode_local, validate_local_ranges

ENDPOINT = "https://api.openai.com/v1/responses"


def validate_request(request: ModelRequest, options: OpenAIProviderConfig | None = None) -> None:
    validate_local_ranges(request)
    if request.generation.get("seed") is not None:
        raise ValueError("OpenAI Responses adapter does not support generation.seed")
    opts = options or OpenAIProviderConfig()
    fps = request.fps or opts.frame_fps
    count = 0
    for video in request.videos:
        assert video.start_s is not None and video.end_s is not None
        count += math.ceil((video.end_s - video.start_s) * fps)
    if count > opts.max_frames:
        raise ValueError("fixed FPS exceeds OpenAI max_frames; adjust explicitly, never silently")


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        model: str,
        options: OpenAIProviderConfig,
        *,
        client: Any = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.options = options
        self.http = JSONHTTP(options.api_key_env, client=client, environ=environ)
        self.media = LocalMedia(options.frame_height, options.media_timeout_s)

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "endpoint": ENDPOINT,
            "adapter_version": 1,
            "httpx_version": importlib.metadata.version("httpx"),
            "imageio_ffmpeg_version": importlib.metadata.version("imageio-ffmpeg"),
            "input_mode": "timestamped_frames_no_audio",
            "options": self.options.model_dump(),
        }

    def close(self) -> None:
        try:
            self.media.close()
        finally:
            self.http.close()

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ProviderResponse:
        from video_report.providers.base import PermanentProviderError

        try:
            validate_request(request, self.options)
        except ValueError as exc:
            raise PermanentProviderError("unsupported_input", str(exc)) from None
        content: list[dict[str, Any]] = [{"type": "input_text", "text": request.user}]
        n_frames = 0
        encoded_bytes = 0
        for video in request.videos:
            fps = request.fps or self.options.frame_fps
            frames = self.media.frames(video, fps, self.options.max_frames)
            n_frames += len(frames)
            for t, path in frames:
                content.append(
                    {
                        "type": "input_text",
                        "text": f"Video {video.video_id}; clip-relative time {t:.3f} seconds.",
                    }
                )
                encoded = encode_local(path, self.options.max_payload_bytes)
                encoded_bytes += len(encoded)
                if encoded_bytes > self.options.max_payload_bytes:
                    raise PermanentProviderError(
                        "payload_limit", "encoded frames exceed payload limit"
                    )
                content.append(
                    {
                        "type": "input_image",
                        "detail": self.options.image_detail,
                        "image_url": "data:image/jpeg;base64," + encoded,
                    }
                )
        if n_frames > self.options.max_frames:
            raise PermanentProviderError("frame_limit", "total frames exceed max_frames")
        body: dict[str, Any] = {
            "model": request.model,
            "input": [{"role": "user", "content": content}],
            "store": False,
            "stream": False,
        }
        if request.system:
            body["instructions"] = request.system
        for key in ("temperature", "top_p", "max_output_tokens"):
            if request.generation.get(key) is not None:
                body[key] = request.generation[key]
        if request.generation.get("response_format") == "json":
            body["text"] = {"format": {"type": "json_object"}}
        data = self.http.post(
            ENDPOINT,
            {"Authorization": f"Bearer {self.http.key}"},
            body,
            timeout_s,
            self.options.max_payload_bytes,
        )
        response = self._to_response(data)
        return replace(
            response,
            metadata={
                "input_mode": "timestamped_frames_no_audio",
                "frames": n_frames,
                "fps": request.fps or self.options.frame_fps,
            },
        )

    @staticmethod
    def _to_response(data: dict[str, Any]) -> ProviderResponse:
        status = optional_string(data.get("status"))
        if status not in {"completed", "incomplete"}:
            raise invalid_response("Responses status must be completed or incomplete")
        output = data.get("output")
        if not isinstance(output, list):
            raise invalid_response("Responses output must be an array")
        text = []
        for item in output:
            if not isinstance(item, dict):
                raise invalid_response("Responses output items must be objects")
            if item.get("type") == "reasoning":
                continue
            if item.get("type") != "message" or not isinstance(item.get("content"), list):
                raise invalid_response("expected an output message with content")
            for part in item["content"]:
                if not isinstance(part, dict):
                    raise invalid_response("message parts must be objects")
                if part.get("type") == "output_text":
                    value = optional_string(part.get("text"))
                    if value is None:
                        raise invalid_response("output_text requires text")
                    text.append(value)
                elif part.get("type") == "refusal":
                    optional_string(part.get("refusal"))
                else:
                    raise invalid_response("unsupported output content type")
        raw = object_or_empty(data.get("usage"))
        usage = None
        if raw:
            completion = tokens(raw, "output_tokens")
            reasoning = tokens(
                object_or_empty(raw.get("output_tokens_details")), "reasoning_tokens"
            )
            if completion is not None and reasoning is not None:
                if reasoning > completion:
                    raise invalid_response("reasoning exceeds total output")
                completion -= reasoning
            usage = Usage(
                input_tokens=tokens(raw, "input_tokens"),
                output_tokens=completion,
                reasoning_tokens=reasoning,
                cached_input_tokens=tokens(
                    object_or_empty(raw.get("input_tokens_details")), "cached_tokens"
                ),
                total_tokens=tokens(raw, "total_tokens"),
            )
        return ProviderResponse(
            text="".join(text) or None,
            usage=usage,
            finish_reason=status,
            model_version=optional_string(data.get("model")),
            response_id=optional_string(data.get("id")),
        )
