"""No-video control for the HD-EPIC Argon batching curve (k = 4/8/16/32), exact baseline calls.

    python scripts/build_hd_epic_argon_novideo.py --per-field PER_FIELD_CSV \
        --out data/hd_epic_argon_novideo_v1

PER_FIELD_CSV is results/hd_epic_q40_v1_four_model_baseline_20261110/per_field.csv (branch
docs/hd-epic-four-model-baseline-20261110). For each k, every Argon call becomes one report
holding the same localisation questions in the same order (one report = one call with
batch_size = k), so the no-video run differs from the baseline only in that no video is sent.
Questions, options and time-range text are taken unchanged from data/hd_epic_q40_v1.
Offline; never invokes a model.
"""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-field", type=Path, required=True)
    ap.add_argument("--src", type=Path, default=Path("data/hd_epic_q40_v1"))
    ap.add_argument("--out", type=Path, default=Path("data/hd_epic_argon_novideo_v1"))
    ap.add_argument("--model", default="argon")
    ap.add_argument("--ks", default="4,8,16,32")
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output already exists")
    reports = {
        r["id"]: r for r in map(json.loads, (args.src / "reports.jsonl").read_text().splitlines())
    }
    gt = {
        (g["report_id"], g["field_id"]): g
        for g in map(json.loads, (args.src / "ground_truth.jsonl").read_text().splitlines())
    }
    rows = [r for r in csv.DictReader(args.per_field.open()) if r["model"] == args.model]
    for k in args.ks.split(","):
        calls: OrderedDict[str, list[dict]] = OrderedDict()
        for r in rows:
            if r["batch_size"] == k:
                calls.setdefault(r["call_id"], []).append(r)
        out_reps, out_gt = [], []
        for call_id, rs in calls.items():
            src = reports[rs[0]["report_id"]]
            by_id = {f["id"]: f for f in src["fields"]}
            fields = [by_id[r["field_id"]] for r in rs]
            for r in rs:  # targets must agree with the baseline export
                assert gt[src["id"], r["field_id"]]["target"] == r["target"], (
                    call_id,
                    r["field_id"],
                )
            rid = f"{src['id']}__k{k}_{call_id.split('::')[1].replace('=', '')}"
            out_reps.append(
                {
                    **src,
                    "id": rid,
                    "fields": fields,
                    "metadata": {
                        **src.get("metadata", {}),
                        "source_report": src["id"],
                        "baseline_call_id": call_id,
                    },
                }
            )
            out_gt += [{**gt[src["id"], f["id"]], "report_id": rid} for f in fields]
        d = args.out / f"k{k}"
        d.mkdir(parents=True)
        (d / "reports.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out_reps))
        (d / "ground_truth.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out_gt))
        man = json.loads((args.src / "manifest.json").read_text())
        man.update(
            benchmark_id=f"hd_epic_argon_novideo_v1_k{k}",
            description=f"HD-EPIC Argon baseline calls at k={k}, one report per call "
            "(no-video control).",
            notes="see scripts/build_hd_epic_argon_novideo.py",
        )
        (d / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
        sizes = {len(r["fields"]) for r in out_reps}
        print(f"k={k}: {len(out_reps)} calls, {len(out_gt)} questions, call sizes {sorted(sizes)}")


if __name__ == "__main__":
    main()
