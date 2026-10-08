"""Parse a saved raw response into per-field answers. Pure function; no I/O.

Expected response: a JSON object whose keys are the requested field ids and whose values are
choice labels (``A``..``Z``, ``AA``..), e.g. ``{"primary_appliance": "B"}``.

Strictness (parser ``json_letters_v1``):
* whole-response failures -> every requested field is ``invalid_output``:
  empty/None text, invalid JSON, duplicate keys (ambiguous), top level not an object
* per field: key absent -> ``missing``; present -> raw value kept for the scorer to validate
* keys not requested are recorded in ``extra_keys``; they do not invalidate requested fields
* surrounding whitespace is ignored. Markdown code fences are rejected unless
  ``allow_markdown_fence=True`` (a separately versioned, explicitly configured leniency).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from video_report.io_utils import DuplicateKeyError, strict_json_loads

PARSER_VERSION = "json_letters_v1"

ResponseStatus = Literal["ok", "empty", "invalid_json", "duplicate_keys", "not_object"]

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*)\n```$", re.DOTALL)


@dataclass(frozen=True)
class ParsedResponse:
    status: ResponseStatus
    answers: dict[str, Any] = field(default_factory=dict)  # requested field id -> raw value
    missing: tuple[str, ...] = ()
    extra_keys: tuple[str, ...] = ()
    detail: str | None = None


def parse_response(
    text: str | None, requested: Sequence[str], *, allow_markdown_fence: bool = False
) -> ParsedResponse:
    if text is None or not text.strip():
        return ParsedResponse("empty", detail="empty response")
    body = text.strip()
    if allow_markdown_fence:
        m = _FENCE_RE.match(body)
        if m:
            body = m.group(1).strip()
    try:
        obj = strict_json_loads(body)
    except DuplicateKeyError as exc:
        return ParsedResponse("duplicate_keys", detail=str(exc))
    except json.JSONDecodeError as exc:
        return ParsedResponse("invalid_json", detail=str(exc))
    if not isinstance(obj, dict):
        return ParsedResponse("not_object", detail=f"top-level JSON is {type(obj).__name__}")
    answers = {k: obj[k] for k in requested if k in obj}
    missing = tuple(k for k in requested if k not in obj)
    extra = tuple(sorted(k for k in obj if k not in set(requested)))
    return ParsedResponse("ok", answers, missing, extra)
