"""Write one experiment config per (benchmark group, batch size) for the CC4D answer-type runs.

Input is the output dir of scripts/build_cc4d_answer_types.py. Each group is run at the batch
sizes the proposal asks for (docs/cc4d_answer_type_proposal.md). With --pilot, every config also
gets a pilot.yaml limited to the reports of those recordings.

    python scripts/make_cc4d_configs.py --data data/cc4d_answer_types_v1 \
        --out experiments/cc4d_answer_types --video-root ~/datasets/captaincook4d/gopro_360p \
        [--model gemini-3.5-flash-lite] [--concurrency 1] [--pilot 1_32 2_30]

Prints the number of calls per config and in total. The prompts are shared, in <out>/prompts/.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import yaml

PRICING = {  # Gemini 3.5 Flash-Lite, global standard tier
    "currency": "USD",
    "input_per_million": 0.3,
    "cached_input_per_million": 0.03,
    "output_per_million": 2.5,
    "source": "https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing"
    " ; global standard Gemini 3.5 Flash-Lite; checked 2026-10-08",
}


def batch_sizes(group):
    """The proposal's k values for one group."""
    if group == "eight_step" or group.endswith("not_done"):
        return [1]
    if m := re.search(r"h(\d+)(_rev)?$", group):
        return [int(m.group(1))]
    if group.startswith("e4"):
        return [8]
    full = [1, 4, 8, 16] if group.startswith("e5_") else [1, 4, 8]
    return full[1:] if group.endswith("_rev") else full  # order does not matter at k = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--pilot", nargs="*", default=[])
    args = ap.parse_args()
    prompts = args.out / "prompts"
    assert (prompts / "user.txt").is_file(), f"missing {prompts}/user.txt"
    total = pilot_total = 0
    for gdir in sorted(p for p in args.data.iterdir() if (p / "manifest.json").is_file()):
        reports = [json.loads(x) for x in (gdir / "reports.jsonl").read_text().splitlines()]
        for k in batch_sizes(gdir.name):
            cdir = args.out / f"{gdir.name}_k{k}"
            cdir.mkdir(parents=True, exist_ok=True)
            rel = lambda p, c=cdir: os.path.relpath(p.resolve(), c.resolve())  # noqa: E731
            cfg = {
                "name": f"cc4d_{gdir.name}_k{k}",
                "description": f"CC4D answer-type experiment, group {gdir.name}, {k} per call.",
                "data": {
                    "benchmark": rel(gdir),
                    "require_local_videos": True,
                    "video_root": os.path.expanduser(args.video_root),
                },
                "method": {"name": "batched", "batch_size": k},
                "prompt": {
                    "system": rel(prompts / "system.txt"),
                    "user": rel(prompts / "user.txt"),
                    "context_keys": ["task_name", "recipe", "error_definitions"],
                },
                "provider": {
                    "name": "gemini",
                    "model": args.model,
                    "gemini": {
                        "backend": "vertex_express",
                        "api_key_env": "GEMINI_API_KEY",
                        "clip_height": 480,
                        "clip_fps": 2.0,
                        # CC4D recordings run up to 39 min; a 480p clip is ~30 KB/s
                        "max_payload_bytes": 200_000_000,
                    },
                },
                "video": {"use_time_range": True, "fps": 2.0},
                "generation": {
                    "temperature": 0.0,
                    "max_output_tokens": 4096,
                    "response_format": "json",
                },
                "execution": {"concurrency": args.concurrency, "max_attempts": 3, "timeout_s": 300},
                "pricing": PRICING if args.model == "gemini-3.5-flash-lite" else None,
            }
            calls = sum(math.ceil(len(r["fields"]) / k) for r in reports)
            (cdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            line = f"{cdir.name:24} {calls:6} calls"
            total += calls
            if args.pilot:
                ids = [
                    r["id"]
                    for r in reports
                    if r["source"]["original_ids"]["recording_id"] in args.pilot
                ]
                if ids:
                    pilot = {
                        **cfg,
                        "name": cfg["name"] + "_pilot",
                        "data": {**cfg["data"], "report_ids": ids},
                    }
                    (cdir / "pilot.yaml").write_text(yaml.safe_dump(pilot, sort_keys=False))
                    n = sum(math.ceil(len(r["fields"]) / k) for r in reports if r["id"] in ids)
                    pilot_total += n
                    line += f"   pilot {n:4}"
            print(line)
    print(f"total {total} calls per model" + (f"; pilot {pilot_total}" if args.pilot else ""))


if __name__ == "__main__":
    main()
