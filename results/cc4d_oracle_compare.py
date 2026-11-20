"""Oracle clips vs the full-video runs on the same step questions (Qwen3-VL-32B, CC4D).

python results/cc4d_oracle_compare.py RUNS_DIR ORACLE_RUN [EVAL_ID ...]
Reads field_scores.jsonl (newest evaluation) of ORACLE_RUN, k32_fwd_k8 and k32_fwd_kall, keeps the step
questions `stepNN_correct` / `stepNN_error_types` of steps that were done (the oracle set), and prints per
field kind: accuracy, default-answer rate (correct = true, error_types = []), and paired differences of the
oracle minus each full-video run, 95% bootstrap interval over recordings.
"""
import collections
import glob
import json
import re
import sys

import numpy as np

runs, oracle = sys.argv[1:3]
rng = np.random.default_rng(0)


def load(run, split_id=False):
    f = sorted(glob.glob(f"{runs}/{run}/evaluations/2*/field_scores.jsonl"))[-1]
    out = {}
    for line in open(f):
        s = json.loads(line)
        m = re.fullmatch(r"step(\d+)_(correct|error_types)", s["field_id"])
        if not m or s["extra"]["not_done_item"]:
            continue
        rep = s["report_id"]
        if split_id:
            rep = rep.split("__")[0]
        out[rep, int(m[1]), m[2]] = (int(s["correct"]), s["predicted"])
    return out


O, A8, AA = load(oracle, True), load("k32_fwd_k8"), load("k32_fwd_kall")
keys = sorted(set(O) & set(A8) & set(AA))
reps = sorted({k[0] for k in keys})
print(f"{len(keys)} step questions, {len(reps)} recordings")
data = {"oracle": O, "k8": A8, "kall": AA}


def default(kind, pred):
    return float(pred is True) if kind == "correct" else float(pred == [] or pred == "[]")


def boot(fn, sub, B=1000):
    by = collections.defaultdict(list)
    for k in sub:
        by[k[0]].append(k)
    rs = sorted(by)
    est = fn([k for r in rs for k in by[r]])
    bs = [fn([k for r in rng.choice(rs, len(rs)) for k in by[r]]) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{est:6.1f} [{lo:6.1f}, {hi:6.1f}]"


for kind in ("correct", "error_types", "both"):
    sub = [k for k in keys if kind == "both" or k[2] == kind]
    print(f"\n== {kind} (n = {len(sub)})")
    for n, d in data.items():
        acc = lambda s, d=d: 100 * np.mean([d[k][0] for k in s])  # noqa: E731
        dfl = lambda s, d=d: 100 * np.mean([default(k[2], d[k][1]) for k in s])  # noqa: E731
        print(f"  {n:7} accuracy {boot(acc, sub)}   default rate {boot(dfl, sub)}")
    for n in ("k8", "kall"):
        diff = lambda s, n=n: 100 * np.mean([O[k][0] - data[n][k][0] for k in s])  # noqa: E731
        print(f"  oracle - {n:4} accuracy {boot(diff, sub)}")
