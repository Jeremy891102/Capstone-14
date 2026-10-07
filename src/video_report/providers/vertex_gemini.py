"""Dennis's global Vertex Express endpoint and x-goog-api-key authentication."""

from __future__ import annotations

import importlib.metadata
import re
from typing import Any

from video_report.config import GeminiProviderConfig
from video_report.providers.base import ModelRequest, ProviderResponse, Usage
from video_report.providers.http_api import (
    JSONHTTP,
    invalid_response,
    object_or_empty,
    optional_string,
    tokens,
)
from video_report.providers.media import LocalMedia, encode_local, validate_local_ranges

BASE = "https://aiplatform.googleapis.com/v1/publishers/google/models/"


def validate_request(request: ModelRequest) -> None:
    validate_local_ranges(request)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", request.model):
        raise ValueError("Vertex model must be a bare model ID, not an OpenRouter slug or URL")


class VertexGeminiProvider:
    name = "gemini"

    def __init__(
        self,
        model: str,
        options: GeminiProviderConfig,
        *,
        client: Any = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.options = options
        self.http = JSONHTTP(options.api_key_env, client=client, environ=environ)
        self.media = LocalMedia(options.clip_height, options.media_timeout_s)

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "backend": "vertex_express",
            "model": self.model,
            "endpoint": BASE + self.model + ":generateContent",
            "adapter_version": 1,
            "httpx_version": importlib.metadata.version("httpx"),
            "imageio_ffmpeg_version": importlib.metadata.version("imageio-ffmpeg"),
            "input_mode": "local_clip_no_audio",
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
            validate_request(request)
        except ValueError as exc:
            raise PermanentProviderError("unsupported_input", str(exc)) from None
        parts: list[dict[str, Any]] = []
        encoded_bytes = 0
        for video in request.videos:
            parts.append({"text": f"Video {video.video_id}; this clip starts at 0 seconds."})
            clip = self.media.clip(video, request.fps or self.options.clip_fps)
            encoded = encode_local(clip, self.options.max_payload_bytes)
            encoded_bytes += len(encoded)
            if encoded_bytes > self.options.max_payload_bytes:
                raise PermanentProviderError("payload_limit", "encoded clips exceed payload limit")
            parts.append({"inlineData": {"mimeType": "video/mp4", "data": encoded}})
        parts.append({"text": request.user})
        generation: dict[str, Any] = {}
        for src, dst in [
            ("temperature", "temperature"),
            ("top_p", "topP"),
            ("max_output_tokens", "maxOutputTokens"),
            ("seed", "seed"),
        ]:
            if request.generation.get(src) is not None:
                generation[dst] = request.generation[src]
        if request.generation.get("response_format") == "json":
            generation["responseMimeType"] = "application/json"
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation,
        }
        if request.system:
            body["systemInstruction"] = {"parts": [{"text": request.system}]}
        data = self.http.post(
            BASE + request.model + ":generateContent",
            {"x-goog-api-key": self.http.key},
            body,
            timeout_s,
            self.options.max_payload_bytes,
        )
        return self._to_response(data)

    @staticmethod
    def _to_response(data: dict[str, Any]) -> ProviderResponse:
        raw = object_or_empty(data.get("usageMetadata"))
        usage = (
            Usage(
                input_tokens=tokens(raw, "promptTokenCount"),
                output_tokens=tokens(raw, "candidatesTokenCount"),
                reasoning_tokens=tokens(raw, "thoughtsTokenCount"),
                cached_input_tokens=tokens(raw, "cachedContentTokenCount"),
                total_tokens=tokens(raw, "totalTokenCount"),
            )
            if raw
            else None
        )
        candidates = data.get("candidates")
        if candidates is None or candidates == []:
            feedback = object_or_empty(data.get("promptFeedback"))
            block = optional_string(feedback.get("blockReason"))
            if not block:
                raise invalid_response("missing candidates without a block reason")
            return ProviderResponse(text=None, usage=usage, metadata={"block_reason": block})
        if not isinstance(candidates, list) or not isinstance(candidates[0], dict):
            raise invalid_response("candidates must be an array of objects")
        candidate = candidates[0]
        finish = optional_string(candidate.get("finishReason"))
        if candidate.get("content") is None and finish in {
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "MAX_TOKENS",
        }:
            return ProviderResponse(
                text=None,
                usage=usage,
                finish_reason=finish,
                metadata={"input_mode": "local_clip_no_audio"},
            )
        content = object_or_empty(candidate.get("content"))
        parts = content.get("parts")
        if not isinstance(parts, list):
            raise invalid_response("candidate parts must be an array")
        text = []
        for part in parts:
            if not isinstance(part, dict):
                raise invalid_response("candidate parts must be objects")
            value = optional_string(part.get("text"))
            if value is not None and not part.get("thought", False):
                text.append(value)
        return ProviderResponse(
            text="".join(text) or None,
            usage=usage,
            finish_reason=optional_string(candidate.get("finishReason")),
            model_version=optional_string(data.get("modelVersion")),
            response_id=optional_string(data.get("responseId")),
            metadata={"input_mode": "local_clip_no_audio"},
        )
