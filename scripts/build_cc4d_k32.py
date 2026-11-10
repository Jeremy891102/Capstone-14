"""CC4D full-report batching experiment (docs/cc4d/cc4d_k32_experiment_design.md).

Reads data/cc4d_dense_v1 and keeps the main experiment's 64 recordings. Per report:
  - skipped_steps is dropped (its options would reveal which steps are missing);
  - every step that was not done (Missing Step) gets the three step questions with target
    "not_visible" (its correct / error_types questions are turned into abstention questions,
    duration is added), at the step's own position;
  - every field gets allow_not_visible.
Writes <out>/fwd and <out>/rev (rev = whole question list flipped), and one config per
(order, k in 8/16/32/all) under <exp>/ plus pilot.yaml for the --pilot recordings.

    python scripts/build_cc4d_k32.py --video-root ~/datasets/captaincook4d/gopro_360p \
        --pilot 23_26 1_32 2_28 12_48 5_27
"""

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path

import yaml

NV = "not_visible"
ALL = 100  # batch_size for "all": above the 83-question maximum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", type=Path, default=Path("data/cc4d_dense_v1"))
    ap.add_argument("--main", type=Path, default=Path("data/cc4d_answer_types_v1/base/reports.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/cc4d_k32_v1"))
    ap.add_argument("--exp", type=Path, default=Path("experiments/cc4d_k32"))
    ap.add_argument("--prompts", type=Path, default=Path("experiments/cc4d_answer_types/prompts"))
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--model", default="gemini-4-argon")
    ap.add_argument("--pilot", nargs="*", default=[])
    args = ap.parse_args()

    rows = lambda p: [json.loads(x) for x in p.read_text().splitlines()]  # noqa: E731
    main_ids = {r["source"]["original_ids"]["recording_id"] for r in rows(args.main)}
    gt = {}
    for g in rows(args.dense / "ground_truth.jsonl"):
        gt[g["report_id"], g["field_id"]] = g
    reports = [r for r in rows(args.dense / "reports.jsonl") if r["source"]["original_ids"]["recording_id"] in main_ids]
    assert len(reports) == 64, len(reports)

    built, n_missing = [], 0
    for r in reports:
        rid = r["id"]
        fields = [f for f in r["fields"] if f["id"] != "skipped_steps"]
        gts = {f["id"]: dict(gt[rid, f["id"]]) for f in fields}
        # steps with no footage: the Missing Step ones
        missing = [
            f for f in fields
            if f["id"].endswith("_correct") and any(e["tag"].startswith("Missing Step") for e in gts[f["id"]]["evidence"]["errors"])
        ]
        n_missing += len(missing)
        for f in missing:
            step = f["id"].removesuffix("_correct")
            assert f"{step}_duration" not in gts
            dur = {
                "id": f"{step}_duration",
                "question": re.sub(r"^Was (the step .*) done correctly\?$", r"How many seconds did \1 take?", f["question"]),
                "answer_type": "seconds",
            }
            assert dur["question"] != f["question"]
            last = max(i for i, x in enumerate(fields) if x["id"].startswith(step + "_"))
            fields.insert(last + 1, dur)  # after the step's other questions
            gts[dur["id"]] = {"report_id": rid, "field_id": dur["id"], "evidence": gts[f["id"]]["evidence"]}
            for x in (f["id"], f"{step}_error_types", dur["id"]):
                if x in gts:
                    gts[x]["target"] = NV
        fields = [{**f, "allow_not_visible": True} for f in fields]
        for v in r["input"]["videos"]:  # the VM stores <rid>.mp4, not <rid>_360p.mp4
            v["uri"] = v["uri"].replace("_360p", "")
        built.append((r, fields, [gts[f["id"]] for f in fields]))
    print(f"{len(built)} reports, {n_missing} steps not done")

    for order in ("fwd", "rev"):
        d = args.out / order
        d.mkdir(parents=True, exist_ok=True)
        out_reports, out_gt = [], []
        for r, fields, truth in built:
            if order == "rev":
                fields, truth = fields[::-1], truth[::-1]
            out_reports.append({**r, "fields": fields, "metadata": {**r["metadata"], "created_by": "build_cc4d_k32", "order": order}})
            out_gt += truth
        (d / "reports.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_reports))
        (d / "ground_truth.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_gt))
        man = json.loads((args.dense / "manifest.json").read_text())
        man.update(benchmark_id=f"cc4d_k32_v1_{order}", description=f"CC4D full-report batching, {order}: 64 recordings, abstention questions for steps not done, every field allows not_visible.")
        (d / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
        print(order, sum(len(r["fields"]) for r in out_reports), "fields")

    prompts = args.exp / "prompts"
    shutil.copytree(args.prompts, prompts, dirs_exist_ok=True)
    cfg0 = {
        "prompt": {"system": "../prompts/system.txt", "user": "../prompts/user.txt", "context_keys": ["task_name", "recipe", "error_definitions"]},
        "provider": {
            "name": "gemini",
            "model": args.model,
            "gemini": {
                "backend": "vertex_express",
                "api_key_env": "GEMINI_API_KEY",
                "clip_height": 480,
                "clip_fps": 2.0,
                "max_payload_bytes": 200_000_000,
                "base_url": "https://aiplatform.googleapis.com/v1/projects/663601952811/locations/global/publishers/google/models/",
            },
        },
        "video": {"use_time_range": True, "fps": 2.0},
        "generation": {"temperature": 0.0, "max_output_tokens": 8192, "response_format": "json"},
        "execution": {"concurrency": 4, "max_attempts": 3, "timeout_s": 600},
        "pricing": {
            "currency": "USD", "input_per_million": 2.0, "cached_input_per_million": 0.1, "output_per_million": 10.0,
            "source": "third-party listings (litellm, dvarahq), introductory Vertex rate to 2026-11-30; not Google's page",
        },
    }
    pilot_ids = [r["id"] for r, _, _ in built if r["source"]["original_ids"]["recording_id"] in args.pilot]
    for order in ("fwd", "rev"):
        for label, k in (("8", 8), ("16", 16), ("32", 32), ("all", ALL)):
            cdir = args.exp / f"{order}_k{label}"
            cdir.mkdir(parents=True, exist_ok=True)
            rel = lambda p, c=cdir: os.path.relpath(p.resolve(), c.resolve())  # noqa: E731
            cfg = {
                "name": f"cc4d_k32_{order}_k{label}",
                "description": f"CC4D full-report batching, {order}, k={label}.",
                "data": {"benchmark": rel(args.out / order), "require_local_videos": True, "video_root": os.path.expanduser(args.video_root)},
                "method": {"name": "batched", "batch_size": k},
                **{**cfg0, "prompt": {kk: (rel(prompts / Path(v).name) if kk != "context_keys" else v) for kk, v in cfg0["prompt"].items()}},
            }
            (cdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            calls = sum(math.ceil(len(f) / k) for _, f, _ in built)
            line = f"{cdir.name:14} {calls:5} calls"
            if pilot_ids:
                pilot = {**cfg, "name": cfg["name"] + "_pilot", "data": {**cfg["data"], "report_ids": pilot_ids}}
                (cdir / "pilot.yaml").write_text(yaml.safe_dump(pilot, sort_keys=False))
                line += f"  pilot {sum(math.ceil(len(f) / k) for r, f, _ in built if r['id'] in pilot_ids)}"
            print(line)


if __name__ == "__main__":
    main()
