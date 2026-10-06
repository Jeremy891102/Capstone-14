"""Gemini adapter (Gemini Developer API via the official ``google-genai`` SDK).

Supported inputs (v1):
* Local video files only, uploaded with the Files API (``client.files.upload``) and polled
  until ``ACTIVE``. Remote URIs (http(s), gs://, YouTube) are rejected at run creation.
* Optional clip offsets from report time ranges -> ``types.VideoMetadata(start_offset,
  end_offset)``; optional ``video.fps`` -> ``VideoMetadata.fps`` (SDK range (0, 24]).
* Text system/user prompts; ``generation.response_format: json`` -> ``response_mime_type``.

Retries: SDK retries are disabled explicitly (``HttpRetryOptions(attempts=1)``) so the runner is
the only retry layer. Note: in google-genai 2.28.0 the resumable-upload chunk loop in
``_api_client.py`` has its own small fixed retry for unfinalized chunks that is not
configurable; it only affects uploads, not generation.

Uploaded files are memoized per provider instance (in memory, keyed by path/size/mtime) so
per-field calls on one video upload it once per process. This is not a persisted inference
cache. Files are deleted on ``close()`` unless ``delete_uploaded_files: false``.

Not live-validated in this repository; see docs/design.md.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path
from typing import Any

from video_report.config import GeminiProviderConfig
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

SUPPORTED_VIDEO_MIME_TYPES = frozenset(
    {
        "video/mp4",
        "video/mpeg",
        "video/mov",
        "video/quicktime",
        "video/avi",
        "video/x-msvideo",
        "video/x-flv",
        "video/mpg",
        "video/webm",
        "video/wmv",
        "video/3gpp",
    }
)
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def validate_request(request: ModelRequest) -> None:
    """Reject inputs this adapter does not implement (called at run creation)."""
    for v in request.videos:
        if not v.is_local:
            raise ValueError(
                f"{request.call_id}: video {v.video_id!r} is a remote URI ({v.location!r}); "
                "the Gemini adapter v1 supports local files only"
            )
        if v.mime_type not in SUPPORTED_VIDEO_MIME_TYPES:
            raise ValueError(
                f"{request.call_id}: video {v.video_id!r} has unsupported or unknown mime type "
                f"{v.mime_type!r}; set video.mime_type in reports.jsonl"
            )


def _offset(seconds: float) -> str:
    return f"{float(seconds)}s"


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        model: str,
        options: GeminiProviderConfig,
        *,
        client: Any = None,
        sleep: Any = time.sleep,
        monotonic: Any = time.monotonic,
        environ: dict[str, str] | None = None,
    ) -> None:
        from google.genai import types  # deferred: mock runs must not need the SDK

        self._types = types
        self.model = model
        self._options = options
        self._sleep = sleep
        self._monotonic = monotonic
        env = os.environ if environ is None else environ
        self._api_key = env.get(options.api_key_env, "")
        if client is None:
            if not self._api_key:
                raise PermanentProviderError(
                    "missing_credentials",
                    f"environment variable {options.api_key_env} is not set",
                )
            from google import genai

            client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
            )
        self._client = client
        self._uploads: dict[tuple[str, int, int], Any] = {}
        self._upload_locks: dict[tuple[str, int, int], threading.Lock] = {}
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- interface

    def describe(self) -> dict[str, Any]:
        try:
            from google.genai import version as genai_version

            sdk = genai_version.__version__
        except Exception:  # pragma: no cover - defensive
            sdk = "unknown"
        return {
            "provider": self.name,
            "model": self.model,
            "sdk": "google-genai",
            "sdk_version": sdk,
            "api_key_env": self._options.api_key_env,
            "sdk_retries": "disabled (HttpRetryOptions(attempts=1))",
        }

    def close(self) -> None:
        if not self._options.delete_uploaded_files:
            return
        with self._lock:
            files = list(self._uploads.values())
            self._uploads.clear()
        for f in files:
            # Best effort; uploaded files also expire server-side.
            with contextlib.suppress(Exception):
                self._client.files.delete(name=f.name)

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ProviderResponse:
        try:
            validate_request(request)
        except ValueError as exc:
            raise PermanentProviderError("unsupported_input", str(exc)) from None
        try:
            parts = [self._video_part(v, timeout_s, request.fps) for v in request.videos]
            parts.append(self._types.Part(text=request.user))
            config = self._generation_config(request, timeout_s)
            response = self._client.models.generate_content(
                model=request.model,
                contents=[self._types.Content(role="user", parts=parts)],
                config=config,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise self._classify(exc) from None
        return self._to_response(response)

    # ----------------------------------------------------------------- helpers

    def _generation_config(self, request: ModelRequest, timeout_s: float) -> Any:
        t = self._types
        g = request.generation
        kwargs: dict[str, Any] = {
            "http_options": t.HttpOptions(
                timeout=int(timeout_s * 1000),  # milliseconds
                retry_options=t.HttpRetryOptions(attempts=1),
            )
        }
        if request.system:
            kwargs["system_instruction"] = request.system
        for key in ("temperature", "top_p", "max_output_tokens", "seed"):
            if g.get(key) is not None:
                kwargs[key] = g[key]
        if g.get("response_format") == "json":
            kwargs["response_mime_type"] = "application/json"
        return t.GenerateContentConfig(**kwargs)

    def _video_part(self, video: VideoInput, timeout_s: float, fps: float | None) -> Any:
        t = self._types
        uploaded = self._upload(video, timeout_s)
        part_kwargs: dict[str, Any] = {
            "file_data": t.FileData(file_uri=uploaded.uri, mime_type=video.mime_type)
        }
        meta: dict[str, Any] = {}
        if video.start_s is not None and video.end_s is not None:
            meta["start_offset"] = _offset(video.start_s)
            meta["end_offset"] = _offset(video.end_s)
        if fps is not None:
            meta["fps"] = fps
        if meta:
            part_kwargs["video_metadata"] = t.VideoMetadata(**meta)
        return t.Part(**part_kwargs)

    def _upload(self, video: VideoInput, timeout_s: float) -> Any:
        path = Path(video.location)
        try:
            st = path.stat()
        except FileNotFoundError:
            raise PermanentProviderError(
                "video_not_found", f"video {video.video_id!r} not found at {path}"
            ) from None
        key = (str(path), st.st_size, st.st_mtime_ns)
        with self._lock:
            if key in self._uploads:
                return self._uploads[key]
            key_lock = self._upload_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                if key in self._uploads:
                    return self._uploads[key]
            f = self._client.files.upload(
                file=str(path),
                config=self._types.UploadFileConfig(
                    mime_type=video.mime_type,
                    display_name=f"video_report:{video.video_id}",
                    http_options=self._types.HttpOptions(timeout=int(timeout_s * 1000)),
                ),
            )
            f = self._wait_active(f)
            with self._lock:
                self._uploads[key] = f
            return f

    def _wait_active(self, f: Any) -> Any:
        deadline = self._monotonic() + self._options.file_active_timeout_s
        while True:
            state = str(getattr(f.state, "name", f.state) or "")
            if state == "ACTIVE":
                return f
            if state == "FAILED":
                raise PermanentProviderError("file_processing_failed", f"file {f.name} FAILED")
            if self._monotonic() >= deadline:
                raise RetryableProviderError(
                    "file_processing_timeout",
                    f"file {f.name} not ACTIVE after {self._options.file_active_timeout_s}s",
                )
            self._sleep(self._options.file_poll_interval_s)
            f = self._client.files.get(name=f.name)

    def _classify(self, exc: Exception) -> ProviderError:
        msg = redact(f"{type(exc).__name__}: {exc}", (self._api_key,))
        try:
            import httpx
        except ImportError:  # pragma: no cover - httpx is a google-genai dependency
            httpx = None  # type: ignore[assignment]
        from google.genai import errors as genai_errors

        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", None)
            if code in _RETRYABLE_STATUS:
                return RetryableProviderError(f"http_{code}", msg, status_code=code)
            return PermanentProviderError(f"http_{code}", msg, status_code=code)
        if httpx is not None:
            # Only connection failures prove the request was never sent.
            if isinstance(exc, httpx.ConnectTimeout):
                return RetryableProviderError("connect_timeout", msg)
            if isinstance(exc, httpx.ConnectError):
                return RetryableProviderError("connect_error", msg)
            if isinstance(exc, httpx.TimeoutException):
                return ProviderTimeoutError("timeout", msg)
            if isinstance(exc, httpx.TransportError):
                # e.g. ReadError / RemoteProtocolError: the connection broke after sending.
                return RetryableProviderError("transport_error", msg, outcome_unknown=True)
        if isinstance(exc, TimeoutError):
            return ProviderTimeoutError("timeout", msg)
        # Unknown exceptions are not retried (retrying a bug only multiplies cost), and may
        # happen after the server already processed the request (e.g. response parsing).
        return PermanentProviderError("unexpected_error", msg, outcome_unknown=True)

    def _to_response(self, response: Any) -> ProviderResponse:
        texts: list[str] = []
        finish_reason = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            cand = candidates[0]
            fr = getattr(cand, "finish_reason", None)
            finish_reason = str(getattr(fr, "name", fr)) if fr is not None else None
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "text", None) and not getattr(part, "thought", False):
                    texts.append(part.text)
        um = getattr(response, "usage_metadata", None)
        usage = None
        if um is not None:
            usage = Usage(
                input_tokens=getattr(um, "prompt_token_count", None),
                output_tokens=getattr(um, "candidates_token_count", None),
                reasoning_tokens=getattr(um, "thoughts_token_count", None),
                cached_input_tokens=getattr(um, "cached_content_token_count", None),
                total_tokens=getattr(um, "total_token_count", None),
            )
        feedback = getattr(response, "prompt_feedback", None)
        block = getattr(feedback, "block_reason", None) if feedback is not None else None
        return ProviderResponse(
            text="".join(texts) if texts else None,
            usage=usage,
            finish_reason=finish_reason,
            model_version=getattr(response, "model_version", None),
            response_id=getattr(response, "response_id", None),
            metadata={"block_reason": str(getattr(block, "name", block))} if block else {},
        )
