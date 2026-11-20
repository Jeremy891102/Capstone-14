"""Oracle-clip variant of data/cc4d_k32_v1/fwd: one report per done step, the clip is that step's span.

    python scripts/build_cc4d_oracle.py --pad 5 --out data/cc4d_oracle_v1

For each step that was done (annotated start/end >= 0) the report keeps the two step questions
(`stepNN_correct`, `stepNN_error_types`, same text and options as in the full report) and a time_range of
[start - pad, end + pad] seconds, so the model sees only the footage of the step it is asked about.
Report id is `<report>__<stepNN>`; ground-truth rows are copied unchanged. Abstention questions (steps not
done) and duration questions are left out (the span would give the duration away). Offline; never invokes a model.
"""

import argparse
import copy
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/cc4d_k32_v1/fwd"))
    ap.add_argument("--out", type=Path, default=Path("data/cc4d_oracle_v1"))
    ap.add_argument("--pad", type=float, default=5.0)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output already exists")
    gt = {}
    for line in (args.src / "ground_truth.jsonl").read_text().splitlines():
        g = json.loads(line)
        gt[g["report_id"], g["field_id"]] = g
    reports, truths = [], []
    for line in (args.src / "reports.jsonl").read_text().splitlines():
        r = json.loads(line)
        tr = r["input"]["videos"][0]["time_range"]
        for f in r["fields"]:
            m = re.fullmatch(r"(step\d+)_correct", f["id"])
            if not m:
                continue
            g = gt[r["id"], f["id"]]
            start, end = g["evidence"]["start_time"], g["evidence"]["end_time"]
            if start < 0:
                continue
            fields = [x for x in r["fields"] if x["id"] in (f"{m[1]}_correct", f"{m[1]}_error_types")]
            new = copy.deepcopy(r)
            new["id"] = f"{r['id']}__{m[1]}"
            new["fields"] = fields
            new["input"]["videos"][0]["time_range"] = {
                **tr,
                "start": round(max(0.0, start - args.pad), 3),
                "end": round(min(tr["end"], end + args.pad), 3),
            }
            new["metadata"] = {**r["metadata"], "oracle_pad": args.pad, "source_report": r["id"]}
            reports.append(new)
            truths += [{**gt[r["id"], x["id"]], "report_id": new["id"]} for x in fields]
    args.out.mkdir(parents=True)
    wr = lambda name, rows: (args.out / name).write_text("".join(json.dumps(x) + "\n" for x in rows))  # noqa: E731
    wr("reports.jsonl", reports)
    wr("ground_truth.jsonl", truths)
    man = json.loads((args.src / "manifest.json").read_text())
    man.update(
        benchmark_id="cc4d_oracle_v1",
        description=f"CC4D oracle clips: one report per done step, step span +/- {args.pad} s.",
        notes="time_range is the annotated step span padded; see scripts/build_cc4d_oracle.py.",
    )
    (args.out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    spans = [r["input"]["videos"][0]["time_range"] for r in reports]
    mean = sum(s["end"] - s["start"] for s in spans) / len(spans)
    print(f"{len(reports)} reports, {len(truths)} fields, mean clip {mean:.0f} s")


if __name__ == "__main__":
    main()
