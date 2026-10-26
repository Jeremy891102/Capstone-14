import hashlib
import json

import pytest

from video_report.providers.media import PreparedClipCache


def test_cache_validates_once_but_rechecks_hash_on_every_use(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"valid original bytes")
    validations = []
    preparations = []

    def prepare():
        preparations.append(True)
        return source

    def validate(p):
        validations.append(p)
        assert p.read_bytes() == source.read_bytes()

    cache = PreparedClipCache(tmp_path / "cache")
    p = cache.get(("video", 0, 3, 2, 480), prepare, validate)
    assert cache.get(("video", 0, 3, 2, 480), prepare, validate) == p
    assert len(preparations) == len(validations) == 1
    p.write_bytes(b"partial file")
    assert cache.get(("video", 0, 3, 2, 480), prepare, validate).read_bytes() == source.read_bytes()
    assert len(preparations) == 2


def test_resume_checks_decode_and_receipt(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"good")
    root = tmp_path / "cache"
    calls = []

    def validate(p):
        calls.append(p)

    p = PreparedClipCache(root).get(("v",), lambda: source, validate)
    receipt = p.with_suffix(".json")
    before = receipt.read_bytes()
    PreparedClipCache(root).get(("v",), lambda: pytest.fail("should reuse valid cache"), validate)
    assert len(calls) == 2 and receipt.read_bytes() == before
    # A matching hash is insufficient when a previously saved file cannot decode.
    p.write_bytes(b"broken")
    receipt.write_text(json.dumps({"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}))

    def decode(path):
        if path.read_bytes() != b"good":
            raise ValueError("decode failed")

    assert PreparedClipCache(root).get(("v",), lambda: source, decode).read_bytes() == b"good"


def test_unknown_cache_and_failed_validation_are_never_published(tmp_path):
    root = tmp_path / "cache"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"bad")

    def reject(p):
        raise ValueError("invalid duration")

    with pytest.raises(ValueError, match="duration"):
        PreparedClipCache(root).get(("v",), lambda: source, reject)
    assert not list(root.glob("*.mp4")) and not list(root.glob("*.json"))


@pytest.mark.parametrize(
    "returncode,duration,frames,expected",
    [
        (0, "00:00:02.00", 4, True),
        (1, "00:00:02.00", 4, False),
        (0, "00:00:00.10", 4, False),
        (0, None, 0, False),
        (0, "00:00:02.00", 0, False),
    ],
)
def test_clip_decoder_rejects_errors_or_wrong_duration(
    monkeypatch, tmp_path, returncode, duration, frames, expected
):
    import sys
    from types import SimpleNamespace

    from video_report.providers.media import validate_prepared_clip

    monkeypatch.setitem(
        sys.modules, "imageio_ffmpeg", SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg")
    )

    def run(args, **kwargs):
        assert "-xerror" in args and "-nostdin" in args and "0:v:0" in args
        return SimpleNamespace(
            returncode=returncode,
            stderr=f"Duration: {duration}, start: 0\nframe= {frames}" if duration else "",
        )

    monkeypatch.setattr("video_report.providers.media.subprocess.run", run)
    if expected:
        validate_prepared_clip(tmp_path / "clip.mp4", 2, 2, 30)
    else:
        with pytest.raises(ValueError):
            validate_prepared_clip(tmp_path / "clip.mp4", 2, 2, 30)
