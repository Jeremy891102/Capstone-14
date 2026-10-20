import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def pilot(monkeypatch, tmp_path, request):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("run_hd_epic_baseline_pilot")
    monkeypatch.setenv("GEMINI_API_KEY", "test-placeholder-before-main")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/pilot.py").write_text("# isolated CLI test\n")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/pilot.py"))
    (tmp_path / ".env").write_text("GEMINI_API_KEY=local-test-not-a-credential\n")
    root = tmp_path / "freeze"
    (root / "requests").mkdir(parents=True)
    sizes = {"E1": 90, "E2": 16, "E3": 8, "E4": 4, "E5": 2, "E6_homogeneous": 2, "E6_mixed": 2}
    if getattr(request, "param", False):
        sizes = dict(module.EXPECTED)
    runs = {}
    hashes = {}
    events = []
    for name, count in sizes.items():
        requests = {}
        for i in range(count):
            data = {"call_id": f"{name}-{i}", "report_id": module.PILOT_IDS[0]}
            if getattr(request, "param", False) and i % 2:
                data["report_id"] = "hdepic-dense-003"
            requests[data["call_id"]] = SimpleNamespace(
                **data,
                system=module.SYSTEM,
                field_ids=("f",),
                videos=[SimpleNamespace(start_s=0, end_s=10)],
                to_json=lambda d=data: d,
            )
        runs[name] = SimpleNamespace(
            load_requests=lambda r=requests: r,
            verify_snapshot=lambda: {},
            load_config=lambda: SimpleNamespace(
                execution=SimpleNamespace(concurrency=1, max_attempts=1)
            ),
        )
        p = root / "requests" / f"{name}.jsonl"
        p.write_text("".join(json.dumps(q.to_json()) + "\n" for q in requests.values()))
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (root / "selection_manifest.json").write_text(
        json.dumps(
            {
                "source_hashes": {},
                "prompt_sha256": hashlib.sha256(module.SYSTEM.encode()).hexdigest(),
                "request_file_hashes": hashes,
            }
        )
    )
    monkeypatch.setattr(module, "create_run", lambda config, *a, **kw: runs[config.parent.name])
    monkeypatch.setattr(module, "verify_derived_ground_truth", lambda *a: "verified-gt")
    monkeypatch.setattr(module, "execute_run", lambda *a, **kw: events.append("execute"))
    monkeypatch.setattr(module.JSONHTTP, "post", lambda *a, **kw: pytest.fail("HTTP forbidden"))
    if not hasattr(module, "_real_verify_media_inputs"):
        module._real_verify_media_inputs = module.verify_media_inputs
    monkeypatch.setattr(module, "verify_media_inputs", lambda *a: {})
    return module, root, events


def test_default_cli_is_offline_and_records_experiment_counts(pilot, monkeypatch):
    module, root, events = pilot
    monkeypatch.setattr(sys, "argv", ["pilot", "--freeze", str(root)])
    module.main()
    plan = json.loads((root / "pilot/pilot_plan.json").read_text())
    assert plan["total_calls"] == 124 and plan["calls"]["E5"] == 2
    assert plan["derived_ground_truth_sha256"]["E1"] == "verified-gt"
    assert not events


def test_execute_budget_failure_precedes_any_inference(pilot, monkeypatch):
    module, root, events = pilot
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--freeze", str(root), "--execute", "--max-estimated-cost-usd", "0"]
    )
    with pytest.raises(ValueError, match="budget"):
        module.main()
    assert not events and not (root / "pilot/pilot_plan.json").exists()


def test_changed_gt_blocks_preflight_before_inference(pilot, monkeypatch):
    module, root, events = pilot

    def reject(*args):
        raise ValueError("derived GT does not match source")

    monkeypatch.setattr(module, "verify_derived_ground_truth", reject)
    monkeypatch.setattr(sys, "argv", ["pilot", "--freeze", str(root), "--execute"])
    with pytest.raises(ValueError, match="source"):
        module.main()
    assert not events


@pytest.mark.parametrize("pilot", [True], indirect=True)
def test_full_scope_freezes_all_4355_calls_without_http(pilot, monkeypatch):
    module, root, events = pilot
    monkeypatch.setattr(sys, "argv", ["runner", "--freeze", str(root), "--scope", "full"])
    module.main()
    plan = json.loads((root / "full/full_plan.json").read_text())
    assert plan["total_calls"] == 4355
    assert plan["calls"] == module.EXPECTED
    assert plan["scope"] == "full"
    assert not events and not (root / "pilot").exists()


@pytest.mark.parametrize("pilot", [True], indirect=True)
def test_full_budget_rejects_before_any_inference(pilot, monkeypatch):
    module, root, events = pilot
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--freeze",
            str(root),
            "--scope",
            "full",
            "--execute",
            "--max-estimated-cost-usd",
            "0",
        ],
    )
    with pytest.raises(ValueError, match="budget"):
        module.main()
    assert not events


@pytest.mark.parametrize("pilot", [True], indirect=True)
def test_full_requires_exact_frozen_requests(pilot, monkeypatch):
    module, root, events = pilot
    path = root / "requests/E5.jsonl"
    rows = path.read_text().splitlines()
    rows.pop()
    path.write_text("\n".join(rows) + "\n")
    manifest = json.loads((root / "selection_manifest.json").read_text())
    manifest["request_file_hashes"]["E5"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "selection_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["runner", "--freeze", str(root), "--scope", "full"])
    with pytest.raises(ValueError, match="inputs differ"):
        module.main()
    assert not events


@pytest.mark.parametrize("pilot", [True], indirect=True)
def test_full_resume_uses_own_existing_runs_and_checks_compatibility(pilot, monkeypatch):
    module, root, events = pilot
    sentinel = root / "pilot/keep.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("historical pilot")
    create = module.create_run
    paths = []
    for name in module.ORDER:
        (root / "full/runs" / name).mkdir(parents=True)
    # Fake create_run expects a config whose parent names the experiment.
    monkeypatch.setattr(module, "RunDir", lambda path: create(path / "config.yaml", None))

    def check(run, **kwargs):
        paths.append(kwargs["config_path"])
        return ["reviewed upgrade"]

    monkeypatch.setattr(module, "check_resume_compatibility", check)
    monkeypatch.setattr(sys, "argv", ["runner", "--freeze", str(root), "--scope", "full"])
    module.main()
    assert len(paths) == 7 and all(p.name == "config.yaml" for p in paths)
    assert sentinel.read_text() == "historical pilot"
    assert not events


@pytest.mark.parametrize("pilot", [True], indirect=True)
def test_full_rejects_matching_but_incomplete_scope(pilot, monkeypatch):
    module, root, events = pilot
    original = module.create_run
    path = root / "requests/E5.jsonl"
    records = [json.loads(x) for x in path.read_text().splitlines()][:-1]
    path.write_text("".join(json.dumps(x) + "\n" for x in records))
    manifest = json.loads((root / "selection_manifest.json").read_text())
    manifest["request_file_hashes"]["E5"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "selection_manifest.json").write_text(json.dumps(manifest))

    def create(config, *args, **kwargs):
        run = original(config, *args, **kwargs)
        if config.parent.name == "E5":
            reqs = run.load_requests()
            reqs.pop(next(reversed(reqs)))
        return run

    monkeypatch.setattr(module, "create_run", create)
    monkeypatch.setattr(sys, "argv", ["runner", "--freeze", str(root), "--scope", "full"])
    with pytest.raises(ValueError, match="wrong full call count"):
        module.main()
    assert not events


@pytest.mark.parametrize("pilot", [True], indirect=True)
@pytest.mark.parametrize("selected", [None, ["E5"]])
def test_full_records_invalid_answers_without_dropping_experiments(pilot, monkeypatch, selected):
    module, root, events = pilot
    original = module.create_run
    http = module.JSONHTTP.post
    generate = module.VertexGeminiProvider.generate
    clip = module.LocalMedia.clip

    def create(config, *args, **kwargs):
        run = original(config, *args, **kwargs)
        run.path = root / "full/runs" / config.parent.name
        run.read_meta = lambda: {"state": "created"}
        return run

    def execute(run, **kwargs):
        events.append((run.path.name, kwargs["retry_failed"]))
        return SimpleNamespace(state="completed")

    def evaluate(path, gt):
        out = root / "evaluations" / path.name
        out.mkdir(parents=True)
        (out / "smoke_test.log").write_text("saved invalid model answer\n")
        (out / "call_parses.jsonl").write_text(
            json.dumps({"parse_status": "invalid_json", "extra_keys": [], "missing": ["f"]}) + "\n"
        )
        return out, {
            "fields": {"field_accuracy": {"value": 0}, "valid_answer_rate": {"value": 0}},
            "usage": {"cost": 0},
        }

    for e in module.ORDER:
        gt = root / "data" / e / "ground_truth.jsonl"
        gt.parent.mkdir(parents=True)
        gt.write_text("test ground truth\n")
    monkeypatch.setattr(
        module, "verify_derived_ground_truth", lambda source, gt, reports: module.sha(gt)
    )
    monkeypatch.setattr(module, "create_run", create)
    monkeypatch.setattr(module, "execute_run", execute)
    monkeypatch.setattr(module, "evaluate_run", evaluate)
    monkeypatch.setattr(
        sys, "argv", ["runner", "--freeze", str(root), "--scope", "full", "--execute"]
    )
    if selected:
        monkeypatch.setattr(sys, "argv", sys.argv + ["--experiments", *selected])
        (root / "full").mkdir(parents=True)
        (root / "full/full_results.json").write_text(
            json.dumps([{"experiment": "E1", "preserved": True}])
        )
    module.main()
    result = json.loads((root / "full/full_results.json").read_text())
    new = [r for r in result if not r.get("preserved")]
    assert len(new) == (len(selected) if selected else 7)
    assert all(not r["all_calls_complete_json"] for r in new)
    if selected:
        assert next(r for r in result if r["experiment"] == "E1")["preserved"]
    assert events == [(e, False) for e in (selected or module.ORDER)]
    assert module.JSONHTTP.post is http
    assert module.VertexGeminiProvider.generate is generate
    assert module.LocalMedia.clip is clip


@pytest.mark.parametrize("pilot", [True], indirect=True)
def test_full_rejects_concurrent_execution_before_inference(pilot, monkeypatch):
    module, root, events = pilot
    original = module.create_run

    def create(config, *args, **kwargs):
        run = original(config, *args, **kwargs)
        run.load_config = lambda: SimpleNamespace(
            execution=SimpleNamespace(concurrency=2, max_attempts=1)
        )
        return run

    monkeypatch.setattr(module, "create_run", create)
    monkeypatch.setattr(
        sys, "argv", ["runner", "--freeze", str(root), "--scope", "full", "--execute"]
    )
    with pytest.raises(ValueError, match="concurrency=1"):
        module.main()
    assert not events


@pytest.mark.parametrize("pilot", [True], indirect=True)
def test_only_e5_can_run_without_touching_other_experiments(pilot, monkeypatch):
    module, root, events = pilot
    monkeypatch.setattr(
        sys, "argv", ["runner", "--freeze", str(root), "--scope", "full", "--experiments", "E5"]
    )
    module.main()
    plan = json.loads((root / "full/full_E5_plan.json").read_text())
    assert plan["total_calls"] == 73 and plan["calls"] == {"E5": 73}
    assert plan["execute_order"] == ["E5"]
    assert not events and not (root / "full/full_plan.json").exists()


def test_video_bytes_must_match_before_reusing_successful_calls(pilot, tmp_path):
    module, root, events = pilot
    video = tmp_path / "video.mp4"
    video.write_bytes(b"original video bytes")
    run = SimpleNamespace(path=tmp_path / "run", read_all_calls=lambda: [])
    run.path.mkdir()
    req = SimpleNamespace(videos=[SimpleNamespace(location=str(video), is_local=True)])
    module._real_verify_media_inputs(run, [req], {})
    run.read_all_calls = lambda: [{"attempts": [{"outcome": "success"}]}]
    module._real_verify_media_inputs(run, [req], {})
    video.write_bytes(b"different video bytes")
    with pytest.raises(ValueError, match="video bytes changed"):
        module._real_verify_media_inputs(run, [req], {})


def test_existing_attempts_without_video_receipt_cannot_be_reused(pilot, tmp_path):
    module, root, events = pilot
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    run = SimpleNamespace(path=tmp_path, read_all_calls=lambda: [{"attempts": [{}]}])
    req = SimpleNamespace(videos=[SimpleNamespace(location=str(video), is_local=True)])
    with pytest.raises(ValueError, match="missing video fingerprint"):
        module._real_verify_media_inputs(run, [req], {})
