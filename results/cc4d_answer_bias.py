"""Do answers drift toward a default as the question sits later or the batch grows? (Qwen3-VL-32B, CC4D k32)

PYTHONPATH=src python results/cc4d_answer_bias.py RUNS_DIR GT_FWD.jsonl [GT_REV.jsonl]
Reads the saved vLLM answers of k32_fwd_k8, k32_fwd_kall, k32_rev_kall. Per step field kind (correct /
error_types / duration), by question position in the call and by k, prints the rate of: answering the default
(correct = true, error_types = [], duration = 1 or 10) and repeating the previous step's answer of the same
kind. The "truth" rows give the same rates for the ground truth, so a rate above it is a bias, not the data.
"""
import collections
import json
import re
import sys
from pathlib import Path

runs = Path(sys.argv[1])
gts = {"fwd": sys.argv[2], "rev": sys.argv[3] if len(sys.argv) > 3 else None}
BINS = [(0, 8), (8, 16), (16, 32), (32, 99)]


def kind(f):
    m = re.fullmatch(r"step\d+_(correct|error_types|duration)", f)
    return m[1] if m else None


def load(run):
    out = []  # (report, call position, step number, kind, value)
    for cf in sorted((runs / run / "calls").glob("*.json")):
        c = json.loads(cf.read_text())
        if c["status"] != "succeeded":
            continue
        n = c["final_attempt"]
        rf = next((runs / run / "responses").glob(f"{cf.stem}/attempt-{n:04d}.json"))
        try:
            vals = json.loads(json.loads(rf.read_text())["response"]["text"])
        except (TypeError, ValueError):
            continue
        for i, f in enumerate(c["field_ids"]):
            if kind(f) and f in vals:
                v = vals[f]
                out.append((c["report_id"], i, int(re.search(r"\d+", f)[0]), kind(f), v))
    return out


def stats(items):
    """items: (report, pos, step, kind, value), grouped by kind and bin -> default rates and repeat rate"""
    prev = {}
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for rep, pos, step, k, v in sorted(items, key=lambda x: (x[0], x[3], x[2])):
        key = (rep, k)
        b = next(j for j, (lo, hi) in enumerate(BINS) if lo <= pos < hi)
        d = {"correct": v is True, "error_types": v == [], "duration": v in (1, 10)}[k]
        res[k][b].append((d, key in prev and prev[key] == v))
        prev[key] = v
    return res


def truth(path):
    return [
        (g["report_id"], 0, int(re.search(r"\d+", g["field_id"])[0]), kind(g["field_id"]), g["target"])
        for g in map(json.loads, open(path))
        if kind(g["field_id"])
    ]


def show(name, items):
    res = stats(items)
    for k in ("correct", "error_types", "duration"):
        cells = []
        for b in range(len(BINS)):
            x = res[k].get(b, [])
            cells.append(f"{100*sum(d for d, _ in x)/len(x):4.0f}/{100*sum(r for _, r in x)/len(x):3.0f} (n={len(x):4d})" if x else "-")
        print(f"{name:11}{k:12}" + "  ".join(f"{c:>20}" for c in cells))


print("cell = default rate / repeat-previous rate (%); columns = position in call", BINS)
tr = truth(gts["fwd"])
tr = [(r, 99 if False else 0, s, k, v) for r, _, s, k, v in tr]
print("-- ground truth (position bin 0 only; same for all positions)")
res = stats(tr)
for k in ("correct", "error_types", "duration"):
    x = [t for b in res[k].values() for t in b]
    print(f"truth      {k:12} default {100*sum(d for d,_ in x)/len(x):.0f}%  repeat {100*sum(r for _,r in x)/len(x):.0f}%")
for run in ("k32_fwd_k8", "k32_fwd_kall", "k32_rev_kall"):
    show(run.replace("k32_", ""), load(run))
