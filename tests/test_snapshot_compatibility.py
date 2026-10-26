import json
from pathlib import Path

import pytest

from video_report.io_utils import fingerprint
from video_report.pipeline import check_resume_compatibility, create_run
from video_report.run_store import RunError

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "experiments/exp001_whole_mcq/config.yaml"


def legacy_run(tmp_path):
    run = create_run(CONFIG, tmp_path, run_id="legacy", environ={})
    cfg = json.loads((run.snapshot / "config.json").read_text())
    cfg["method"].pop("batch_size")
    (run.snapshot / "config.json").write_text(json.dumps(cfg))
    # Model a run written before batch_size existed, not a change to a real saved run.
    view = {k: v for k, v in cfg.items() if k not in {"execution", "description", "pricing"}}
    fps = run.load_fingerprints()
    fps["config"] = fingerprint(view)
    fps["overall"] = fingerprint({k: v for k, v in fps.items() if k != "overall"})
    (run.snapshot / "fingerprints.json").write_text(json.dumps(fps))
    return run


def test_legacy_snapshot_and_resume_config_verify_without_rewriting(tmp_path):
    run = legacy_run(tmp_path)
    before = {p: p.read_bytes() for p in run.snapshot.rglob("*") if p.is_file()}
    run.verify_snapshot()
    check_resume_compatibility(
        run, config_path=CONFIG, video_root=None, allow_code_change=False, environ={}
    )
    assert all(p.read_bytes() == value for p, value in before.items())


def test_legacy_compatibility_does_not_accept_real_config_edits(tmp_path):
    run = legacy_run(tmp_path)
    p = run.snapshot / "config.json"
    cfg = json.loads(p.read_text())
    cfg["generation"]["temperature"] = 0.75
    p.write_text(json.dumps(cfg))
    with pytest.raises(RunError, match="modified"):
        run.verify_snapshot()


def test_current_snapshot_requires_its_original_explicit_defaults(tmp_path):
    run = create_run(CONFIG, tmp_path, run_id="current", environ={})
    run.verify_snapshot()
    p = run.snapshot / "config.json"
    cfg = json.loads(p.read_text())
    cfg["method"].pop("batch_size")
    p.write_text(json.dumps(cfg))
    with pytest.raises(RunError, match="modified"):
        run.verify_snapshot()
