"""Rebuild summary.json and print a table from the merged field scores (offline)."""

import collections
import glob
import json
import os

import merged

here = merged.here


def cost(d):
    m = json.load(open(sorted(glob.glob(d + "/evaluations/2*/metrics.json"))[-1]))
    return m["usage"]["cost"], m["execution"]["execution_completion"]["denominator"]


rows = []
for run in merged.runs():
    f = merged.load(run)
    rr = f"{here}/runs_rerun/rr_{run.removeprefix('fl_')}"
    c0, n0 = cost(f"{here}/runs/{run}")
    c1, n1 = cost(rr) if os.path.isdir(rr) else (0, 0)
    t = collections.defaultdict(list)
    for r in f:
        t[r["extra"]["answer_type"]].append(r["correct"] is True)
    nd = [r for r in f if r["extra"].get("not_done_item")]
    rows.append(
        {
            "run": run,
            "fields": len(f),
            "accuracy": sum(r["correct"] is True for r in f) / len(f),
            "valid_answer_rate": 1 - sum(r["outcome"].startswith("invalid") for r in f) / len(f),
            "by_answer_type": {k: sum(v) / len(v) for k, v in t.items()},
            "hallucination": sum(r["extra"].get("hallucination") is True for r in nd) / len(nd) if nd else None,
            "calls_original": n0,
            "calls_rerun": n1,
            "cost_usd_original": c0,
            "cost_usd_rerun": c1,
        }
    )
json.dump(rows, open(f"{here}/summary.json", "w"), indent=1)
print(f"{'run':28}{'fields':>8}{'acc':>7}{'valid':>7}{'rerun calls':>12}")
for r in rows:
    print(f"{r['run']:28}{r['fields']:>8}{r['accuracy']:>7.3f}{r['valid_answer_rate']:>7.3f}{r['calls_rerun']:>12}")
print("cost original", round(sum(r["cost_usd_original"] for r in rows), 2), "rerun", round(sum(r["cost_usd_rerun"] for r in rows), 2))
