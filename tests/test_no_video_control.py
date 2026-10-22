import hashlib
import importlib
import json
from pathlib import Path

import pytest

from video_report.providers.base import ProviderResponse, RetryableProviderError, Usage


@pytest.fixture
def nv(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_no_video_control")


class FakeProvider:
    """Answers every question with its target letter; never touches the network."""

    def __init__(self, nv, flaky: int = 0):
        self.calls = []
        self.flaky = flaky
        self.targets = {}
        for e in nv.EXPERIMENTS:
            for t in nv.read_jsonl_gz(nv.GROUND_TRUTH / f"{e}.jsonl.gz"):
                self.targets[(t["report_id"], t["field_id"])] = t["target"]

    def generate(self, request, *, timeout_s):
        self.calls.append(request)
        if self.flaky:
            self.flaky -= 1
            raise RetryableProviderError("rate_limited", "try again")
        answers = {f: self.targets[(request.report_id, f)] for f in request.field_ids}
        return ProviderResponse(text=json.dumps(answers), usage=Usage(input_tokens=400, output_tokens=12))


def baseline_digest(nv):
    h = hashlib.sha256()
    for p in sorted(nv.BASELINE.rglob("*")):
        if p.is_file():
            h.update(str(p).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_only_the_video_is_removed(nv):
    for e in nv.EXPERIMENTS:
        frozen = nv.load_requests(e)[0]
        request = nv.without_video(frozen).to_json()
        assert frozen["videos"] and request["videos"] == []
        assert {k: v for k, v in request.items() if k != "videos"} == {
            k: v for k, v in frozen.items() if k != "videos"
        }


def test_refuses_to_write_into_the_baseline(nv):
    with pytest.raises(SystemExit):
        nv.check_out_dir(nv.BASELINE / "data")
    with pytest.raises(SystemExit):
        nv.check_out_dir(nv.REPO / "results" / "anything")


def test_dry_run_needs_no_key_and_writes_nothing(nv, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert nv.main(["--out", str(tmp_path / "out")]) == 0
    assert not (tmp_path / "out").exists()
    assert "dry run only" in capsys.readouterr().out


def test_run_resume_and_score_leave_baseline_untouched(nv, tmp_path):
    before = baseline_digest(nv)
    out = tmp_path / "out"
    provider = FakeProvider(nv, flaky=1)
    nv.main(["--out", str(out), "--limit", "3", "--execute"], provider=provider)
    first = len(provider.calls)
    assert first == 3 * 3 + 1  # three experiments, one retried call
    assert all(r.videos == () for r in provider.calls)

    nv.main(["--out", str(out), "--limit", "3", "--execute"], provider=provider)
    assert len(provider.calls) == first  # resumed: nothing re-sent

    nv.main(["--out", str(out), "--score"])
    rows = list((out / "per_field_no_video.csv").read_text().splitlines())
    assert len(rows) - 1 == 3 + 3 * 8 + 3 * 32
    assert all(line.split(",")[9] == "True" for line in rows[1:])  # correct_no_video
    assert json.loads((out / "summary.json").read_text())["by_batch_size"]
    assert baseline_digest(nv) == before


def test_cost_cap_stops_new_calls(nv, tmp_path):
    provider = FakeProvider(nv)
    nv.main(
        ["--out", str(tmp_path), "--experiments", "E1", "--limit", "50", "--execute",
         "--max-cost-usd", "0.0005", "--concurrency", "1"],
        provider=provider,
    )
    assert 0 < len(provider.calls) < 50
