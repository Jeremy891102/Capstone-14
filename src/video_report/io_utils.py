"""Small, dependency-free helpers for JSON handling, hashing, and atomic file writes."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DuplicateKeyError(ValueError):
    """Raised when a JSON object contains the same key more than once."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateKeyError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def strict_json_loads(text: str) -> Any:
    """Parse JSON, rejecting duplicate object keys (the stdlib silently keeps the last one)."""
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for fingerprints: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def iter_jsonl(path: Path) -> Iterator[tuple[int, Any]]:
    """Yield (line_number, parsed_object) for each non-blank line, with strict key handling."""
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield lineno, strict_json_loads(line)
            except (json.JSONDecodeError, DuplicateKeyError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON line: {exc}") from exc


def _fsync_dir(directory: Path) -> None:
    # Best effort: makes the rename durable on POSIX; not supported on every platform.
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write text so readers see either the old file or the complete new file, never a prefix.

    Writes to a unique temporary file in the same directory, fsyncs it, then os.replace()s it
    over the destination (atomic on POSIX and Windows for same-filesystem renames).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
    _fsync_dir(path.parent)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[Any]) -> None:
    atomic_write_text(
        path, "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    )


def read_json(path: Path) -> Any:
    return strict_json_loads(path.read_text(encoding="utf-8"))
