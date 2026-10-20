import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def pilot(monkeypatch, tmp_path):
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
    runs = {}
    hashes = {}
    events = []
    for name, count in sizes.items():
        requests = {}
        for i in range(count):
            data = {"call_id": f"{name}-{i}", "report_id": module.PILOT_IDS[0]}
            requests[data["call_id"]] = SimpleNamespace(
                **data,
                system=module.SYSTEM,
                field_ids=("f",),
                videos=[SimpleNamespace(start_s=0, end_s=10)],
                to_json=lambda d=data: d,
            )
        runs[name] = SimpleNamespace(load_requests=lambda r=requests: r, verify_snapshot=lambda: {})
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
