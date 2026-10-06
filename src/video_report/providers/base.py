"""Provider interface: perform ONE generation attempt; never retry.

Retry, backoff, and scheduling live only in ``video_report.runner``. Provider adapters translate
SDK exceptions into the three error classes below so the runner can decide what to do.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

# --------------------------------------------------------------------------- requests


@dataclass(frozen=True)
class VideoInput:
    video_id: str
    location: str  # absolute local path, or remote URI
    is_local: bool
    mime_type: str | None
    start_s: float | None = None  # clip start, seconds from video start (None = whole video)
    end_s: float | None = None


@dataclass(frozen=True)
class ModelRequest:
    """Exactly what a provider receives. Built only from model-facing data.

    ``report_id``/``field_ids`` are identifiers (the mock provider uses them to look up canned
    answers); they carry no target information.
    """

    call_id: str
    report_id: str
    field_ids: tuple[str, ...]
    system: str | None
    user: str
    videos: tuple[VideoInput, ...]
    model: str
    generation: dict[str, Any]
    fps: float | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["field_ids"] = list(self.field_ids)
        d["videos"] = [asdict(v) for v in self.videos]
        return d

    @staticmethod
    def from_json(obj: dict[str, Any]) -> ModelRequest:
        return ModelRequest(
            call_id=obj["call_id"],
            report_id=obj["report_id"],
            field_ids=tuple(obj["field_ids"]),
            system=obj["system"],
            user=obj["user"],
            videos=tuple(VideoInput(**v) for v in obj["videos"]),
            model=obj["model"],
            generation=dict(obj["generation"]),
            fps=obj.get("fps"),
        )


# --------------------------------------------------------------------------- responses


@dataclass(frozen=True)
class Usage:
    """Token usage as reported by the provider. ``None`` means unknown, never zero."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    total_tokens: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderResponse:
    text: str | None  # None when the provider returned no text (e.g. blocked)
    usage: Usage | None
    finish_reason: str | None = None
    model_version: str | None = None
    response_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "usage": self.usage.to_json() if self.usage else None,
            "finish_reason": self.finish_reason,
            "model_version": self.model_version,
            "response_id": self.response_id,
            "metadata": self.metadata,
        }


# --------------------------------------------------------------------------- errors


class ProviderError(Exception):
    """Base class. ``message`` must already be safe to persist (see ``redact``)."""

    retryable: bool = False
    # True when the request may have reached the provider and been processed (and billed)
    # even though we got no usable response, e.g. a read timeout.
    outcome_unknown: bool = False

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        status_code: int | None = None,
        outcome_unknown: bool | None = None,
    ) -> None:
        self.kind = kind
        self.message = redact(message)
        super().__init__(f"{kind}: {self.message}")
        self.status_code = status_code
        if outcome_unknown is not None:
            self.outcome_unknown = outcome_unknown

    def to_json(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "kind": self.kind,
            "message": self.message,
            "status_code": self.status_code,
            "retryable": self.retryable,
            "outcome_unknown": self.outcome_unknown,
        }


class RetryableProviderError(ProviderError):
    """Transient: rate limit, 5xx, connection reset. The request was not processed."""

    retryable = True


class ProviderTimeoutError(RetryableProviderError):
    """No response within the timeout. The remote call may still have run and been billed."""

    outcome_unknown = True


class PermanentProviderError(ProviderError):
    """Will not succeed on retry: bad request, auth, unsupported input, missing file."""

    retryable = False


class Provider(Protocol):
    name: str
    model: str

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ProviderResponse:
        """One attempt. Raise a ``ProviderError`` subclass on failure. Must be thread-safe."""
        ...

    def describe(self) -> dict[str, Any]:
        """Static facts recorded in the run snapshot (SDK version, etc.). No secrets."""
        ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- redaction

_REDACTIONS = [
    # Google API keys
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "[REDACTED_API_KEY]"),
    # key=... / api_key=... / token=... style parameters and headers
    (
        re.compile(
            r"(?i)\b(key|api[_-]?key|x-goog-api-key|access_token|token|authorization)"
            # optional closing quote of the key, separator, optional opening quote of value
            r"([\"']?\s*[=:]\s*[\"']?)(bearer\s+)?[^\s&\"',;}]+"
        ),
        r"\1\2[REDACTED]",
    ),
    # Query strings of URLs (signed URLs carry credentials there)
    (re.compile(r"(https?://[^\s?\"']+)\?[^\s\"']*"), r"\1?[REDACTED_QUERY]"),
]


def redact(text: str, extra_secrets: tuple[str, ...] = ()) -> str:
    """Remove credentials and signed-URL query strings from text before it is persisted."""
    for secret in extra_secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text
