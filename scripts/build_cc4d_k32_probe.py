"""CC4D batch size x position grid (follow-up to cc4d_k32, docs/cc4d/new_experiment/q7.zh.md 7.7).

Per report, 2 probe sets of 4 questions. Each set is asked in 9 calls: batch size N in {8, 16, 32}
x probe block at the start / middle / end of the call. The other N - 4 questions are fillers: the same
random sequence for the 3 positions of one N, nested across N, so only the probes' position differs.
Each call is written as its own report (id <report>__<set>_n<N>_<pos>) with the fields in call order,
so the existing `batched` method with batch_size 32 makes exactly one call per report.

    python scripts/build_cc4d_k32_probe.py --video-root ~/datasets/captaincook4d/gopro_360p --pilot 23_26 1_32 2_28 12_48 5_27
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import yaml

NS, POS, P = (8, 16, 32), ("start", "mid", "end"), 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/cc4d_k32_v1/fwd"))
    ap.add_argument("--out", type=Path, default=Path("data/cc4d_k32_probe_v1"))
    ap.add_argument("--exp", type=Path, default=Path("experiments/cc4d_k32_probe"))
    ap.add_argument("--base-config", type=Path, default=Path("experiments/cc4d_k32/fwd_k8/config.yaml"))
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--pilot", nargs="*", default=[])
    args = ap.parse_args()

    rows = lambda p: [json.loads(x) for x in p.read_text().splitlines()]  # noqa: E731
    gt = {(g["report_id"], g["field_id"]): g for g in rows(args.src / "ground_truth.jsonl")}
    reports, truth = [], []
    for r in rows(args.src / "reports.jsonl"):
        rng = random.Random(f"probe:{r['id']}")
        fields = {f["id"]: f for f in r["fields"]}
        # probes: step questions that are really asked (not abstention), none shared between the 2 sets
        cand = [f["id"] for f in r["fields"] if re.match(r"step\d+_", f["id"]) and gt[r["id"], f["id"]]["target"] != "not_visible"]
        rng.shuffle(cand)
        for s, probes in enumerate((cand[:P], cand[P : 2 * P])):
            rest = [i for i in fields if i not in probes]
            rng.shuffle(rest)  # filler order; N takes the first N - P (nested)
            for n in NS:
                n = min(n, len(fields))
                fill = rest[: n - P]
                at = {"start": 0, "mid": (n - P) // 2, "end": n - P}
                for pos in POS:
                    order = fill[: at[pos]] + probes + fill[at[pos] :]
                    rid = f"{r['id']}__s{s}_n{n}_{pos}"
                    reports.append({**r, "id": rid, "fields": [fields[i] for i in order],
                                    "metadata": {**r["metadata"], "created_by": "build_cc4d_k32_probe", "probe_set": s, "n": n, "pos": pos, "probes": probes}})
                    truth += [{**gt[r["id"], i], "report_id": rid} for i in order]
    args.out.mkdir(parents=True, exist_ok=True)
    w = lambda name, xs: (args.out / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in xs))  # noqa: E731
    w("reports.jsonl", reports)
    w("ground_truth.jsonl", truth)
    man = json.loads((args.src / "manifest.json").read_text())
    man.update(benchmark_id=args.out.name, description="CC4D batch size x position grid: 2 probe sets of 4 questions per report, N in 8/16/32, probes at start/mid/end.")
    (args.out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(len(reports), "calls,", len(truth), "fields")

    cfg = yaml.safe_load(args.base_config.read_text())
    args.exp.mkdir(parents=True, exist_ok=True)
    rel = lambda p: os.path.relpath(p.resolve(), args.exp.resolve())  # noqa: E731
    base_dir = args.base_config.parent
    cfg["name"], cfg["description"] = "cc4d_k32_probe", "CC4D batch size x position grid, one call per report."
    cfg["data"]["benchmark"] = rel(args.out)
    cfg["data"]["video_root"] = os.path.expanduser(args.video_root)
    cfg["method"]["batch_size"] = max(NS)
    cfg["prompt"] = {k: (rel((base_dir / v).resolve()) if k != "context_keys" else v) for k, v in cfg["prompt"].items()}
    (args.exp / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    ids = [r["id"] for r in reports if r["source"]["original_ids"]["recording_id"] in args.pilot]
    if ids:
        (args.exp / "pilot.yaml").write_text(yaml.safe_dump({**cfg, "name": "cc4d_k32_probe_pilot", "data": {**cfg["data"], "report_ids": ids}}, sort_keys=False))
        print("pilot", len(ids), "calls")


if __name__ == "__main__":
    main()
