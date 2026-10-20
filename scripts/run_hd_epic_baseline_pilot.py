#!/usr/bin/env python3
"""Prepare or execute ONLY the two-report pilot from an immutable baseline freeze.

Default is offline. Paid calls require --execute. Successful calls are resumed, not repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from prepare_hd_epic_baseline import PILOT_IDS, SYSTEM, dump, sha
from video_report.benchmarks.baseline import validate_pilot_budget, verify_derived_ground_truth
from video_report.evaluation.evaluate import evaluate_run
from video_report.pipeline import check_resume_compatibility, create_run, execute_run
from video_report.providers.base import PermanentProviderError
from video_report.providers.http_api import JSONHTTP
from video_report.providers.media import LocalMedia, PreparedClipCache, validate_prepared_clip
from video_report.providers.vertex_gemini import VertexGeminiProvider
from video_report.run_store import RunDir, call_key

ORDER = ["E5", "E1", "E2", "E3", "E4", "E6_homogeneous", "E6_mixed"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-code-change",
        action="store_true",
        help="Allow reviewed package changes; frozen model inputs must still match",
    )
    parser.add_argument("--max-estimated-cost-usd", type=float, default=1.50)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    freeze = args.freeze.resolve()
    root = freeze / "pilot"
    env = {}
    for line in (repo / ".env").read_text().splitlines():
        name, sep, value = line.removeprefix("export ").partition("=")
        if sep and name.strip() == "GEMINI_API_KEY":
            parts = shlex.split(value, comments=True)
            if len(parts) != 1 or not parts[0]:
                raise SystemExit("Invalid local credential; no request sent.")
            env[name.strip()] = parts[0]
    if not env.get("GEMINI_API_KEY"):
        raise SystemExit("Missing local credential; no request sent.")
    os.environ.update(env)
    manifest = json.loads((freeze / "selection_manifest.json").read_text())
    for name, digest in manifest["source_hashes"].items():
        require(sha(repo / "data/hd_epic_q40_v1" / name) == digest, "source data changed")
    require(
        manifest["prompt_sha256"] == hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "system prompt changed",
    )
    runs = {}
    costs = {}
    plan = {}
    hashes = {}
    gt_hashes = {}
    resume_notes = {}
    for e in ORDER:
        require(
            sha(freeze / "requests" / f"{e}.jsonl") == manifest["request_file_hashes"][e],
            "frozen requests changed",
        )
        gt_hashes[e] = verify_derived_ground_truth(
            repo / "data/hd_epic_q40_v1/ground_truth.jsonl",
            freeze / "data" / e / "ground_truth.jsonl",
            freeze / "data" / e / "reports.jsonl",
        )
        resume_notes[e] = []
        config = freeze / "experiments" / e / "pilot.yaml"
        path = root / "runs" / e
        if path.exists():
            run = RunDir(path)
            resume_notes[e] = check_resume_compatibility(
                run,
                config_path=config,
                video_root=None,
                allow_code_change=args.allow_code_change,
                environ=env,
            )
        else:
            run = create_run(config, root / "runs", run_id=e, environ=env)
        reqs = list(run.load_requests().values())
        require(
            all(q.report_id in PILOT_IDS and q.system == SYSTEM for q in reqs),
            "request outside pilot scope",
        )
        full = [
            json.loads(x) for x in (freeze / "requests" / f"{e}.jsonl").read_text().splitlines()
        ]
        require(
            [q.to_json() for q in reqs] == [q for q in full if q["report_id"] in PILOT_IDS],
            "pilot inputs differ from freeze",
        )
        hashes[e] = run.verify_snapshot()
        plan[e] = len(reqs)
        it = sum(
            64 * (q.videos[0].end_s - q.videos[0].start_s) + 114 + 254.375 * len(q.field_ids)
            for q in reqs
        )
        ot = sum(1.2857142857 + 13.1142857143 * len(q.field_ids) for q in reqs)
        costs[e] = (it * 0.3 + ot * 2.5) / 1e6
        runs[e] = run
    estimate = sum(costs.values())
    validate_pilot_budget(sum(plan.values()), estimate, args.max_estimated_cost_usd)
    dump(
        root / "pilot_plan.json",
        {
            "pilot_ids": PILOT_IDS,
            "calls": plan,
            "total_calls": 124,
            "estimate_usd": estimate,
            "estimated_cost_by_experiment": costs,
            "estimate_limit_usd": args.max_estimated_cost_usd,
            "note": "Estimate guard, not a provider billing cap; one attempt, no automatic retries",
            "execute_order": ORDER,
            "snapshot_fingerprints": hashes,
            "derived_ground_truth_sha256": gt_hashes,
            "runner_sha256": sha(Path(__file__)),
        },
    )
    print(f"Pilot frozen: {sum(plan.values())} calls; estimated US${estimate:.6f}", flush=True)
    if not args.execute:
        print("Offline only: no API requests.", flush=True)
        return
    original_generate = VertexGeminiProvider.generate
    original_clip = LocalMedia.clip
    original_post = JSONHTTP.post
    active = {}
    clip_store = PreparedClipCache(root / "media" / "verified-v1")
    total_new = 0
    (root / "media").mkdir(exist_ok=True)

    def generate(self, request, *, timeout_s):
        active["request"] = request
        return original_generate(self, request, timeout_s=timeout_s)

    def clip(self, video, fps):
        start = time.monotonic()
        key = (*self._key(video, fps), self.height, "h264-crf30-veryfast-v1")
        try:
            target = clip_store.get(
                key,
                lambda: original_clip(self, video, fps),
                lambda p: validate_prepared_clip(
                    p, video.end_s - video.start_s, fps, self.timeout_s
                ),
            )
        except ValueError as exc:
            raise PermanentProviderError(
                "invalid_media", "prepared video validation failed"
            ) from exc
        digest = sha(target)
        req = active["request"]
        active["timing"]["media"].append(
            {
                "call_id": req.call_id,
                "clip_sha256": digest,
                "clip_path": str(target),
                "bytes": target.stat().st_size,
                "preprocessing_or_cache_lookup_s": time.monotonic() - start,
            }
        )
        return target

    def post(self, url, headers, body, timeout_s, max_bytes):
        nonlocal total_new
        req = active["request"]
        e = active["experiment"]
        timing = active["timing"]
        run = runs[e]
        key = call_key(req.call_id)
        # Never send anything outside the frozen pilot, even if an adapter is changed.
        require(
            req.to_json() == run.load_requests()[req.call_id].to_json(),
            "request changed before HTTP",
        )
        require(
            total_new < 124 and timing["new_api_attempts"] < plan[e], "pilot API call limit reached"
        )
        payload = json.dumps(body, ensure_ascii=False)
        require(len(payload.encode()) <= max_bytes, "API payload limit exceeded")
        (active["directory"] / "wire_inputs" / f"{key}.json").write_text(payload + "\n")
        total_new += 1
        timing["new_api_attempts"] += 1
        entry = {
            "call_id": req.call_id,
            "report_id": req.report_id,
            "field_ids": list(req.field_ids),
            "payload_bytes": len(payload.encode()),
        }
        timing["api"].append(entry)
        print(f"{e} {total_new}/124: {req.call_id}", flush=True)
        start = time.monotonic()
        try:
            response = original_post(self, url, headers, body, timeout_s, max_bytes)
            (active["directory"] / "raw_outputs" / f"{key}.json").write_text(
                json.dumps(response, ensure_ascii=False).replace(
                    env["GEMINI_API_KEY"], "[REDACTED]"
                )
                + "\n"
            )
            entry["http_success"] = True
            return response
        finally:
            entry["api_round_trip_s"] = time.monotonic() - start
            dump(active["directory"] / "timing.json", timing)

    VertexGeminiProvider.generate = generate
    LocalMedia.clip = clip
    JSONHTTP.post = post
    results = []
    for e in ORDER:
        run = runs[e]
        directory = root / e
        for name in ["wire_inputs", "raw_outputs"]:
            (directory / name).mkdir(parents=True, exist_ok=True)
        timing = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "new_api_attempts": 0,
            "api": [],
            "media": [],
        }
        # Preserve earlier timing records on resume.
        old = directory / "timing.json"
        if old.exists():
            shutil.copy2(old, directory / f"timing_previous_{time.time_ns()}.json")
        active.update(experiment=e, timing=timing, directory=directory)
        start = time.monotonic()
        try:
            mode = "resume" if run.read_meta()["state"] != "created" else "create"
            result = execute_run(
                run,
                mode=mode,
                retry_failed=False,
                session_notes=resume_notes[e],
                on_call_done=lambda c: print("  " + c["status"], flush=True),
            )
            timing["state"] = result.state
        finally:
            timing["execution_wall_s"] = time.monotonic() - start
            dump(directory / "timing.json", timing)
        require(
            sha(freeze / "data" / e / "ground_truth.jsonl") == gt_hashes[e],
            "derived ground truth changed during inference",
        )
        evaluation, metrics = evaluate_run(run.path, freeze / "data" / e / "ground_truth.jsonl")
        shutil.copy2(evaluation / "smoke_test.log", directory / "smoke_test.log")
        dump(directory / "metrics.json", metrics)
        parses = [
            json.loads(x) for x in (evaluation / "call_parses.jsonl").read_text().splitlines()
        ]
        good = all(
            p["parse_status"] == "ok" and not p["extra_keys"] and not p["missing"] for p in parses
        )
        summary = {
            "experiment": e,
            "run_dir": str(run.path),
            "evaluation_dir": str(evaluation),
            "metrics": metrics,
            "timing": timing,
            "all_calls_complete_json": good,
        }
        dump(directory / "summary.json", summary)
        results.append(summary)
        dump(root / "pilot_results.json", results)
        print(
            f"{e}: {metrics['fields']['field_accuracy']}; cost ${metrics['usage']['cost']}",
            flush=True,
        )
        require(
            result.state == "completed" and good,
            "Pilot failed: inspect saved evidence before continuing",
        )
        require(
            metrics["fields"]["valid_answer_rate"]["value"] == 1, "pilot returned invalid answers"
        )
    print("TWO-REPORT PILOT COMPLETE — full baseline has NOT been executed.", flush=True)


if __name__ == "__main__":
    main()
