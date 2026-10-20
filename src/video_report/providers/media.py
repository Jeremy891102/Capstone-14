"""Local visual-only clip/frame preparation for the new direct API adapters."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from video_report.io_utils import atomic_write_json, fingerprint, sha256_file
from video_report.providers.base import ModelRequest, PermanentProviderError, VideoInput

PREPARED_CLIP_VERSION = "h264-crf30-veryfast-output-duration-v2"


def validate_local_ranges(request: ModelRequest) -> None:
    if request.fps is not None and (not math.isfinite(request.fps) or not 0 < request.fps <= 24):
        raise ValueError("FPS must be finite and in (0, 24]")
    for video in request.videos:
        if not video.is_local or video.start_s is None or video.end_s is None:
            raise ValueError("local video files with explicit time_range are required")
        if not math.isfinite(video.start_s) or not math.isfinite(video.end_s):
            raise ValueError("video offsets must be finite")
        if not 0 <= video.start_s < video.end_s:
            raise ValueError("video offsets must satisfy 0 <= start < end")


class LocalMedia:
    """One process-local cache; no persisted short-video dataset, no model calls."""

    def __init__(self, height: int, timeout_s: float) -> None:
        self.height = height
        self.timeout_s = timeout_s
        self.root = Path(tempfile.mkdtemp(prefix="video_report_media_"))
        self._clips: dict[tuple[object, ...], Path] = {}
        self._frames: dict[tuple[object, ...], list[tuple[float, Path]]] = {}
        self._lock = threading.Lock()

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _key(self, video: VideoInput, fps: float) -> tuple[object, ...]:
        path = Path(video.location)
        if not path.is_file():
            raise PermanentProviderError("video_not_found", "local video file is missing")
        try:
            stat = path.stat()
        except OSError:
            raise PermanentProviderError(
                "media_read_failed", "cannot inspect local video"
            ) from None
        return (str(path), stat.st_size, stat.st_mtime_ns, video.start_s, video.end_s, fps)

    def _run(self, args: list[str]) -> None:
        import imageio_ffmpeg

        try:
            result = subprocess.run(
                [str(imageio_ffmpeg.get_ffmpeg_exe()), "-y", "-loglevel", "error", *args],
                capture_output=True,
                timeout=self.timeout_s,
            )
        except OSError:
            raise PermanentProviderError("media_failed", "cannot start local ffmpeg") from None
        except subprocess.TimeoutExpired:
            raise PermanentProviderError("media_timeout", "local ffmpeg timed out") from None
        if result.returncode:
            raise PermanentProviderError("media_failed", "ffmpeg could not prepare the video")

    def clip(self, video: VideoInput, fps: float) -> Path:
        with self._lock:
            key = self._key(video, fps)
            if key not in self._clips:
                assert video.start_s is not None and video.end_s is not None
                out = self.root / f"clip-{len(self._clips)}.mp4"
                self._run(
                    [
                        "-ss",
                        str(video.start_s),
                        "-i",
                        video.location,
                        "-t",
                        str(video.end_s - video.start_s),
                        "-vf",
                        f"scale=-2:{self.height}",
                        "-r",
                        str(fps),
                        "-an",
                        "-c:v",
                        "libx264",
                        "-crf",
                        "30",
                        "-preset",
                        "veryfast",
                        str(out),
                    ]
                )
                self._clips[key] = out
            return self._clips[key]

    def frames(self, video: VideoInput, fps: float, max_frames: int) -> list[tuple[float, Path]]:
        assert video.start_s is not None and video.end_s is not None
        if math.ceil((video.end_s - video.start_s) * fps) > max_frames:
            raise PermanentProviderError(
                "frame_limit", "fixed FPS would exceed max_frames; adjust settings explicitly"
            )
        with self._lock:
            key = (*self._key(video, fps), max_frames)
            if key not in self._frames:
                folder = self.root / f"frames-{len(self._frames)}"
                folder.mkdir()
                self._run(
                    [
                        "-ss",
                        str(video.start_s),
                        "-t",
                        str(video.end_s - video.start_s),
                        "-i",
                        video.location,
                        "-vf",
                        f"fps={fps}:start_time=0,scale=-2:{self.height}",
                        "-frames:v",
                        str(max_frames + 1),
                        "-q:v",
                        "3",
                        str(folder / "%06d.jpg"),
                    ]
                )
                paths = sorted(folder.glob("*.jpg"))
                if not paths or len(paths) > max_frames:
                    raise PermanentProviderError("frame_limit", "empty or oversized frame sequence")
                # Fixed-rate sample positions on the clip-relative presentation timeline.
                self._frames[key] = [(i / fps, p) for i, p in enumerate(paths)]
            return self._frames[key]


def encode_local(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError:
        raise PermanentProviderError("media_read_failed", "cannot read prepared media") from None
    if not raw or len(raw) > max_bytes:
        raise PermanentProviderError("media_size", "empty media or payload limit exceeded")
    return base64.b64encode(raw).decode("ascii")


class PreparedClipCache:
    """Persist validated clips atomically; reject changed or unsealed cached files."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._validated: dict[Path, str] = {}
        self._lock = threading.Lock()

    def get(
        self,
        key: tuple[object, ...],
        prepare: Callable[[], Path],
        validate: Callable[[Path], None],
    ) -> Path:
        with self._lock:
            target = self.root / f"{fingerprint(key)}.mp4"
            receipt = target.with_suffix(".json")
            try:
                record = json.loads(receipt.read_text())
                digest = sha256_file(target)
                if isinstance(record, dict) and record.get("sha256") == digest:
                    if self._validated.get(target) != digest:
                        validate(target)
                        self._validated[target] = digest
                    return target
            except (OSError, ValueError):
                pass  # Missing, partial or changed cache: reconstruct from the source.
            fd, name = tempfile.mkstemp(prefix=".preparing-", suffix=".mp4", dir=self.root)
            os.close(fd)
            temporary = Path(name)
            try:
                shutil.copyfile(prepare(), temporary)
                validate(temporary)
                digest = sha256_file(temporary)
                os.replace(temporary, target)
                atomic_write_json(receipt, {"sha256": digest})
                self._validated[target] = digest
                return target
            finally:
                temporary.unlink(missing_ok=True)


def validate_prepared_clip(path: Path, duration_s: float, fps: float, timeout_s: float) -> None:
    """Fully decode the video and check its duration before publishing/reusing a clip."""
    import imageio_ffmpeg

    try:
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-nostdin",
                "-xerror",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-an",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("prepared clip could not be validated") from exc
    match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", result.stderr)
    frames = re.findall(r"frame=\s*(\d+)", result.stderr)
    if result.returncode or not match or not frames or int(frames[-1]) == 0:
        raise ValueError("prepared clip does not decode as video")
    duration = int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])
    if abs(duration - duration_s) > max(0.25, 2 / fps):
        raise ValueError("prepared clip duration does not match the requested window")
