"""Answer->video attention vs question position, and vs correctness (Qwen3-VL-32B, k = all).

python results/cc4d_attention.py ATTN_DIR ROWS.json.gz [SUFFIX ...]
ATTN_DIR holds k32_fwd_kall{SUFFIX}.jsonl and k32_rev_kall{SUFFIX}.jsonl (scripts/qwen_attention.py
output); SUFFIX is "" (class A, <= 11 min) or "_B". ROWS is rows_qwen32.json.gz (cc4d_k32_rows.py).

Every effect is within question: the same (report, field) is asked forward and reversed, so it
sits at two positions. Slopes use question fixed effects; intervals bootstrap over reports.
Attention measures, per answer (mean over text layers and heads):
  raw    share of all attention on video tokens
  prompt video / (video + prompt text after the video); drops the attention-sink share (system
         prompt, chat header) and the share on earlier answers, which grows with position by
         construction, so `raw` falls with position even if the model looks at the video the same
  late   `prompt` for the last quarter of layers, where almost all video attention sits
"""

import collections
import gzip
import json
import sys

import numpy as np

attn_dir, rows_path, *suffixes = sys.argv[1:]
suffixes = suffixes or [""]
rng = np.random.default_rng(0)

truth = {
    (r[0], r[2], r[3]): (r[4], r[5], r[6])  # order, report, field -> type, correct, not-done item
    for r in json.load(gzip.open(rows_path, "rt"))
    if r[1] == "all"
}
recs = []
for suf in suffixes:
    for order in ("fwd", "rev"):
        for line in open(f"{attn_dir}/k32_{order}_kall{suf}.jsonl"):
            a = json.loads(line)
            t, ok, nd = truth[order, a["report_id"], a["field_id"]]
            x = np.array(a["attention"])  # layers x (video, before video, rest of prompt, answers)
            nl = len(x) * 3 // 4
            share = lambda m: (m[:, 0] / (m[:, 0] + m[:, 2])).mean()  # noqa: E731
            recs.append(
                dict(
                    rep=a["report_id"],
                    q=a["report_id"] + a["field_id"],
                    order=order,
                    type=t,
                    ok=ok,
                    nd=nd,
                    idx=a["index"],
                    n=a["call_len"],
                    dur=a["recorded_input_tokens"],
                    raw=x[:, 0].mean() * 100,
                    prompt=share(x) * 100,
                    late=share(x[nl:]) * 100,
                    sink=x[:, 1].mean() * 100,
                    ans=x[:, 3].mean() * 100,
                    layer=x[:, 0] / (x[:, 0] + x[:, 2]) * 100,
                )
            )
print(f"{len(recs)} answers, {len({r['rep'] for r in recs})} reports, suffixes {suffixes}")

reps = sorted({r["rep"] for r in recs})
by_rep = collections.defaultdict(list)
for i, r in enumerate(recs):
    by_rep[r["rep"]].append(i)
qid = {q: i for i, q in enumerate(sorted({r["q"] for r in recs}))}
Q = np.array([qid[r["q"]] for r in recs])
col = lambda k: np.array([r[k] for r in recs], float)  # noqa: E731


def demean(v, I):
    q = Q[I]
    s = np.bincount(q, v[I], len(qid))
    c = np.bincount(q, minlength=len(qid))
    return v[I] - (s / np.maximum(c, 1))[q]


def slope(x, y, I):
    xd, yd = demean(x, I), demean(y, I)
    return (xd @ yd) / (xd @ xd)


def boot(fn, mask=None, B=300):
    """(estimate, lo, hi) of fn(I) over all rows (or rows where mask), resampling reports."""
    keep = np.ones(len(recs), bool) if mask is None else mask
    sel = lambda ix: np.array([i for i in ix if keep[i]], int)  # noqa: E731
    est = fn(sel(range(len(recs))))
    bs = [
        fn(sel(np.concatenate([by_rep[r] for r in rng.choice(reps, len(reps))]))) for _ in range(B)
    ]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


fmt = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"  # noqa: E731
idx, ok = col("idx"), col("ok")
lg = np.log2(idx + 1)
normal = col("nd") == 0
types = sorted({r["type"] for r in recs})
tmask = {t: np.array([r["type"] == t for r in recs]) for t in types}

print("\n== 1. Attention to the video vs position (percentage points, within question)")
print("   slope per doubling of the question's position (log2(index + 1)), and per +10 positions")
print(
    f"   mean level: raw {col('raw').mean():.1f}%  prompt {col('prompt').mean():.1f}%  "
    f"late {col('late').mean():.1f}%  sink {col('sink').mean():.1f}%  earlier answers "
    f"{col('ans').mean():.1f}%"
)
for m in ("raw", "prompt", "late", "sink", "ans"):
    y = col(m)
    print(
        f"  {m:7s} per doubling {fmt(boot(lambda I: slope(lg, y, I)))}   "
        f"per +10 {fmt(boot(lambda I: slope(idx / 10, y, I)))}"
    )

print("\n== 2. `prompt` slope per doubling, by answer type (n answers)")
for t in types:
    y = col("prompt")
    print(f"  {t:14s} {int(tmask[t].sum()):5d}  {fmt(boot(lambda I: slope(lg, y, I), tmask[t]))}")

print("\n== 3. `prompt` by position bin, minus the question's own mean (pp)")
yd = demean(col("prompt"), np.arange(len(recs)))
for lo, hi in ((0, 8), (8, 16), (16, 32), (32, 48), (48, 99)):
    m = (idx >= lo) & (idx < hi)
    print(f"  index {lo:2d}-{min(hi - 1, 79):2d}  n={int(m.sum()):4d}  {yd[m].mean():+.3f}")

print("\n== 4. `prompt` slope per doubling, by layer quarter (is it the late layers?)")
L = np.array([r["layer"] for r in recs])
q4 = np.array_split(np.arange(L.shape[1]), 4)
for i, ls in enumerate(q4):
    y = L[:, ls].mean(1)
    print(
        f"  layers {ls[0]:2d}-{ls[-1]:2d}  mean {y.mean():5.2f}%  "
        f"{fmt(boot(lambda I: slope(lg, y, I)))}"
    )

print("\n== 5. Does more attention to the video go with a correct answer? (ordinary questions)")
print("   change in P(correct), in pp, per +1 pp of attention (within question, fwd vs rev)")
for m in ("prompt", "late"):
    y = col(m)
    print(f"  {m:7s} all {fmt(boot(lambda I: slope(y, ok * 100, I), normal))}")
    for t in types:
        if t in ("int", "single_choice"):
            continue
        print(f"    {t:14s} {fmt(boot(lambda I: slope(y, ok * 100, I), normal & tmask[t]))}")


def slope_ctrl(x, c, y, I):
    """Coefficient of x on y within question, holding c (position) fixed."""
    X = np.column_stack([demean(x, I), demean(c, I)])
    return np.linalg.lstsq(X, demean(y, I), rcond=None)[0][0]


print("\n== 5b. Same, holding the question's position fixed (is it just position?)")
for m in ("prompt", "late"):
    y = col(m)
    print(f"  {m:7s} all {fmt(boot(lambda I: slope_ctrl(y, lg, ok * 100, I), normal))}")

print("\n== 6. Same question, forward vs reversed: accuracy and attention (ordinary questions)")
for order in ("fwd", "rev"):
    m = normal & np.array([r["order"] == order for r in recs])
    print(f"  {order}  accuracy {ok[m].mean() * 100:.1f}%  prompt {col('prompt')[m].mean():.2f}%")
print(
    "  accuracy slope per doubling of position:",
    fmt(boot(lambda I: slope(lg, ok * 100, I), normal)),
)
