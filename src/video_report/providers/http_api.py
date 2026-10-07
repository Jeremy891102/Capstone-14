"""One HTTP attempt, fixed endpoint, classified failures and safe response decoding."""

from __future__ import annotations

import json
import os
from typing import Any

from video_report.providers.base import (
    PermanentProviderError,
    ProviderTimeoutError,
    RetryableProviderError,
)


class JSONHTTP:
    def __init__(
        self, api_key_env: str, *, client: Any = None, environ: dict[str, str] | None = None
    ) -> None:
        import httpx

        env = os.environ if environ is None else environ
        self.key = env.get(api_key_env, "").strip()
        if client is None and not self.key:
            raise PermanentProviderError("missing_credentials", f"set {api_key_env}")
        self.client = (
            client
            if client is not None
            else httpx.Client(
                transport=httpx.HTTPTransport(retries=0), follow_redirects=False, trust_env=False
            )
        )

    def close(self) -> None:
        self.client.close()

    def post(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_s: float,
        max_bytes: int,
    ) -> dict[str, Any]:
        import httpx

        if len(json.dumps(body, ensure_ascii=False).encode()) > max_bytes:
            raise PermanentProviderError("payload_limit", "request exceeds max_payload_bytes")
        try:
            resp = self.client.post(url, headers=headers, json=body, timeout=timeout_s)
        except (httpx.ConnectTimeout, httpx.ConnectError):
            raise RetryableProviderError(
                "connect_error", "could not establish connection"
            ) from None
        except httpx.TimeoutException:
            raise ProviderTimeoutError("timeout", "remote outcome unknown") from None
        except httpx.TransportError:
            raise RetryableProviderError(
                "transport_error", "remote outcome unknown", outcome_unknown=True
            ) from None
        except Exception:
            raise PermanentProviderError(
                "unexpected_error", "HTTP client failure", outcome_unknown=True
            ) from None
        if resp.status_code != 200:
            code = resp.status_code
            cls = (
                RetryableProviderError
                if code in {408, 429} or code >= 500
                else PermanentProviderError
            )
            raise cls(
                f"http_{code}",
                f"HTTP {code}",
                status_code=code,
                outcome_unknown=code == 408 or code >= 500,
            )
        try:
            data = resp.json()
        except ValueError:
            raise invalid_response("response is not valid JSON") from None
        if not isinstance(data, dict) or data.get("error") is not None:
            raise invalid_response("response must be a success object")
        return data


def invalid_response(detail: str) -> PermanentProviderError:
    return PermanentProviderError("invalid_response", detail, status_code=200, outcome_unknown=True)


def tokens(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is not None and (type(value) is not int or value < 0):
        raise invalid_response("usage counts must be nonnegative integers")
    return value


def object_or_empty(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise invalid_response("usage/details must be objects")
    return raw


def optional_string(raw: Any) -> str | None:
    if raw is not None and not isinstance(raw, str):
        raise invalid_response("response text/identifiers must be strings or null")
    return raw
