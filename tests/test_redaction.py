from __future__ import annotations

import pytest

from video_report.providers.base import PermanentProviderError, redact


@pytest.mark.parametrize(
    "text, secret",
    [
        ("failed with key AIzaSyA1234567890abcdefghijklmnopqrstu", "AIzaSyA1234567890"),
        ("GET https://x.googleapis.com/v1/m?key=abc123secret&alt=json", "abc123secret"),
        (
            "https://storage.googleapis.com/b/o.mp4?X-Goog-Signature=deadbeef&X-Goog-Expires=9",
            "deadbeef",
        ),
        ('{"api_key": "sk-supersecret"}', "sk-supersecret"),
        ("Authorization: Bearer tok_123456", "tok_123456"),
        ("x-goog-api-key: hunter2", "hunter2"),
        ("{'authorization': 'Bearer ya29.a0AfH6SMBx'}", "ya29.a0AfH6SMBx"),
        ("{'api_key': 'plainsecret42'}", "plainsecret42"),
        ("headers={'x-goog-api-key': 'notAIzaShaped99'}", "notAIzaShaped99"),
    ],
)
def test_redact(text: str, secret: str) -> None:
    out = redact(text)
    assert secret not in out and "REDACTED" in out


def test_extra_secret_and_error_message_redacted() -> None:
    assert "s3cr3t" not in redact("value s3cr3t here", ("s3cr3t",))
    err = PermanentProviderError("x", "url https://h/p?sig=abc")
    assert "sig=abc" not in err.message and "sig=abc" not in str(err.to_json())
    assert "sig=abc" not in str(err) and "sig=abc" not in repr(err)
