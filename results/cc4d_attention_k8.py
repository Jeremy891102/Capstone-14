"""Answer->video attention at k = 8 vs k = all, same question (Qwen3-VL-32B, forward order).

PYTHONPATH=src python results/cc4d_attention_k8.py ATTN_DIR ROWS.json.gz RUNS_DIR
ATTN_DIR: k32_fwd_k8.jsonl, k32_fwd_kall.jsonl (+ k32_fwd_kall_B.jsonl is not used; k = 8 covers class A
only). ROWS: rows_qwen32.json.gz. RUNS_DIR: holds the vLLM runs k32_fwd_k8 and k32_fwd_kall (their saved
answers decide which questions got the same answer in both).

Columns are the attention measures of cc4d_attention.py (raw, prompt, late; mean over heads and layers,
`late` = last quarter of layers). Differences are k = 8 minus k = all in percentage points, bootstrapped
over reports. "same position" = the first 8 questions of a report: same index in both prompts, so only the
number of other questions in the prompt differs. "same answer" keeps questions whose saved answer is
identical at k = 8 and k = all, so the attention difference cannot come from a different answer.
"""

import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from video_report.run_store import call_key

attn_dir, rows_path, runs_dir = sys.argv[1:4]
rng = np.random.default_rng(0)
MEASURES = ("raw", "prompt", "late")


def load_attention(path):
    out = {}
    for line in open(path):
        a = json.loads(line)
        x = np.array(a["attention"])
        nl = len(x) * 3 // 4
        share = lambda m: (m[:, 0] / (m[:, 0] + m[:, 2])).mean() * 100  # noqa: E731
        out[a["report_id"], a["field_id"]] = dict(
            idx=a["index"], raw=x[:, 0].mean() * 100, prompt=share(x), late=share(x[nl:])
        )
    return out


def load_answers(run):
    run = Path(runs_dir) / run
    out = {}
    for line in (run / "snapshot/requests.jsonl").read_text().splitlines():
        r = json.loads(line)
        state = json.loads((run / "calls" / f"{call_key(r['call_id'])}.json").read_text())
        if state["status"] != "succeeded":
            continue
        n = state["final_attempt"]
        resp = json.loads(
            (run / f"responses/{call_key(r['call_id'])}/attempt-{n:04d}.json").read_text()
        )
        try:
            vals = json.loads(resp["response"]["text"])
        except (TypeError, ValueError):
            continue
        for f in r["field_ids"]:
            if f in vals:
                v = vals[f]
                out[r["report_id"], f] = sorted(v) if isinstance(v, list) else v
    return out


truth = {
    (r[1], r[2], r[3]): r  # k, report, field -> type, correct, not-done item at r[4:7]
    for r in json.load(gzip.open(rows_path, "rt"))
    if r[0] == "fwd" and r[1] in ("8", "all")
}
k8 = load_attention(f"{attn_dir}/k32_fwd_k8.jsonl")
al = load_attention(f"{attn_dir}/k32_fwd_kall.jsonl")
a8, aa = load_answers("k32_fwd_k8"), load_answers("k32_fwd_kall")
qs = [q for q in k8 if q in al and q in a8 and q in aa]
typ = {q: truth["8", *q][4] for q in qs}
same = {q: a8[q] == aa[q] for q in qs}
first = [q for q in qs if al[q]["idx"] == k8[q]["idx"] < 8]
reps = sorted({q[0] for q in qs})
print(f"{len(qs)} questions, {len(reps)} reports; same position: {len(first)}")
print(f"identical answer at k = 8 and k = all: {sum(same.values())} of {len(qs)}")


def boot(fn, sub, B=500):
    by = collections.defaultdict(list)
    for q in sub:
        by[q[0]].append(q)
    rs = sorted(by)
    if len(rs) < 3:
        return "n/a"
    est = fn([q for r in rs for q in by[r]])
    bs = [fn([q for r in rng.choice(rs, len(rs)) for q in by[r]]) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{est:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]"


def dmean(m):
    return lambda sub: np.mean([k8[q][m] - al[q][m] for q in sub])


def block(title, sub):
    print(f"\n== {title} (n = {len(sub)})")
    print("   " + " " * 18 + "".join(f"{m:>26s}" for m in MEASURES))
    groups = [("all types", sub)] + [
        (t, [q for q in sub if typ[q] == t]) for t in ("bool", "multi_choice", "seconds")
    ]
    for name, g in groups:
        print(
            f"   {name:14s}{len(g):4d}  " + "".join(f"{boot(dmean(m), g):>26s}" for m in MEASURES)
        )


block("1. same position (first 8 questions), all answers", first)
block("2. same position, identical answer in both", [q for q in first if same[q]])
block("3. all matched questions, all answers", qs)
block("4. all matched questions, identical answer in both", [q for q in qs if same[q]])

print("\n== 5. accuracy on ordinary questions, k = 8 vs k = all (pp)")
for name, sub in (("same position", first), ("all matched", qs)):
    sub = [q for q in sub if truth["8", *q][6] == 0]
    ok = lambda k: np.array([truth[k, *q][5] for q in sub]) * 100  # noqa: E731
    d = ok("8") - ok("all")
    by = collections.defaultdict(list)
    for q, x in zip(sub, d):
        by[q[0]].append(x)
    rs = sorted(by)
    bs = [np.mean([x for r in rng.choice(rs, len(rs)) for x in by[r]]) for _ in range(500)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(
        f"   {name:14s} n={len(sub):4d}  k=8 {ok('8').mean():.1f}%  k=all {ok('all').mean():.1f}%"
        f"  diff {d.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]"
    )
