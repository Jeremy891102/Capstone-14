"""CC4D reference-context x video control (docs/cc4d/new_experiment/reference_context_design.zh.md).

Reads data/cc4d_k32_v1/fwd. Per recording, samples --per-type target questions of each of
correct / error_types / duration (from full 8-question batches, abstention questions excluded).
The 7 other questions of the target's k=8 batch are its reference questions.

Writes two benchmarks under <out>, one report per target (same video, one field):
  single/  the target alone
  ctx7/    the target; its 7 references are rendered into context["reference_questions"]
           (unanswered), so the runner needs no change
plus <out>/targets.json (frozen ids), and one config per (model, video, condition) under <exp>.

    python scripts/build_cc4d_reference.py
"""

import argparse
import copy
import json
import os
import random
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from video_report.benchmarks.schema import FieldSpec  # noqa: E402
from video_report.prompting import render_questions  # noqa: E402

K = 8
TYPES = ("correct", "error_types", "duration")
HEADER = (
    "Reference questions about the same video. They are shown only as context: do NOT answer them. "
    "They are questions, not verified statements about the video. Answer only the question(s) "
    "asked below under 'Answer all'.\n\n"
)


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def dump(p, items):
    Path(p).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/cc4d_k32_v1/fwd"))
    ap.add_argument("--out", type=Path, default=Path("data/cc4d_ref_v1"))
    ap.add_argument("--exp", type=Path, default=Path("experiments/cc4d_ref"))
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--no-nv", action="store_true", help="same targets, but no question allows not_visible (writes *_nonv dirs)")
    args = ap.parse_args()
    if args.no_nv:
        args.out, args.exp = args.out.with_name(args.out.name + "_nonv"), args.exp.with_name(args.exp.name + "_nonv")

    gt = {(g["report_id"], g["field_id"]): g for g in rows(args.src / "ground_truth.jsonl")}
    rng = random.Random(args.seed)
    out = {c: ([], []) for c in ("single", "ctx7")}
    targets = []
    for r in rows(args.src / "reports.jsonl"):
        fields = r["fields"]
        full = len(fields) // K * K
        pool = {t: [] for t in TYPES}
        for i, f in enumerate(fields[:full]):
            m = re.fullmatch(r"step\d+_(correct|error_types|duration)", f["id"])
            if m and gt[r["id"], f["id"]]["target"] != "not_visible":
                pool[m.group(1)].append(i)
        for t in TYPES:
            for i in sorted(rng.sample(pool[t], min(args.per_type, len(pool[t])))):
                f = {**fields[i], "allow_not_visible": False} if args.no_nv else fields[i]
                b = i // K * K
                refs = [{**x, "allow_not_visible": False} if args.no_nv else x for j, x in enumerate(fields[b : b + K], start=b) if j != i]
                assert len(refs) == K - 1
                block = render_questions([FieldSpec(**x) for x in refs])
                block = re.sub(r"(?m)^Q(\d+)\. \[field_id: [^\]]*\]", r"R\1.", block)
                rid = f"{r['id']}__{f['id']}"
                targets.append({"report_id": rid, "recording": r["id"], "field_id": f["id"], "type": t,
                                "batch_start": b, "reference_ids": [x["id"] for x in refs]})
                g = {**gt[r["id"], f["id"]], "report_id": rid}
                for cond in out:
                    ctx = dict(r["input"]["context"])
                    if cond == "ctx7":
                        ctx["reference_questions"] = HEADER + block
                    rep = {**r, "id": rid, "fields": [f], "input": {**r["input"], "context": ctx},
                           "metadata": {**r["metadata"], "created_by": "build_cc4d_reference", "condition": cond}}
                    out[cond][0].append(rep)
                    out[cond][1].append(g)

    man = json.loads((args.src / "manifest.json").read_text())
    for cond, (reps, gts) in out.items():
        d = args.out / cond
        d.mkdir(parents=True, exist_ok=True)
        dump(d / "reports.jsonl", reps)
        dump(d / "ground_truth.jsonl", gts)
        (d / "manifest.json").write_text(json.dumps({**man, "benchmark_id": f"{args.out.name}_{cond}",
            "description": f"CC4D reference-context control, {cond}: one target question per report."}, indent=2) + "\n")
    (args.out / "targets.json").write_text(json.dumps({"seed": args.seed, "per_type": args.per_type, "targets": targets}, indent=1))
    print(len(targets), "targets;", {t: sum(x["type"] == t for x in targets) for t in TYPES})

    # model templates: the existing k=8 fwd configs (same provider, video and generation settings)
    base = {"argon": Path("experiments/cc4d_k32/fwd_k8"), "qwen": Path("experiments/cc4d_k32_qwen/fwd_k8")}
    for model, bdir in base.items():
        if not (bdir / "config.yaml").exists():  # this machine only has the other model's template
            continue
        cfg0 = yaml.safe_load((bdir / "config.yaml").read_text())
        for video in ("video", "novideo"):
            for cond in out:
                name = f"{model}_{video}_{cond}"
                cdir = args.exp / name
                cdir.mkdir(parents=True, exist_ok=True)
                rel = lambda p, c=cdir: os.path.relpath(Path(p).resolve(), c.resolve())  # noqa: E731
                cfg = copy.deepcopy(cfg0)
                cfg["name"] = f"{args.exp.name}_{name}"
                cfg["description"] = f"CC4D reference-context control: {model}, {video}, {cond}."
                cfg["data"]["benchmark"] = rel(args.out / cond)
                cfg["method"] = {"name": "whole_report"}
                for k in ("system", "user"):
                    cfg["prompt"][k] = rel(bdir / cfg0["prompt"][k])
                if cond == "ctx7":
                    cfg["prompt"]["context_keys"] = [*cfg["prompt"]["context_keys"], "reference_questions"]
                cfg["video"] = {**cfg["video"], **({"enabled": False} if video == "novideo" else {})}
                (cdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("configs:", sorted(p.name for p in args.exp.iterdir()))


if __name__ == "__main__":
    main()
