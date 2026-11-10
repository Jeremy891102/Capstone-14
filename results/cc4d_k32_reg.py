"""CC4D full-report batching, part 2 (docs/cc4d/new_experiment/q7.zh.md 7.7, 7.8).
7.7 batch size and position separated: linear probability model with question fixed effects,
    y = b1*log2(call length) + b2*log2(questions after it + 1) + b3*reversed, bootstrap over recordings.
7.8 baseline-normalized accuracy as in Q1-Q6: (acc - base) / (1 - base), base = best constant answer.
Reads results/cc4d_k32_v1_gemini4_argon_20261109/rows.json.gz and data/cc4d_k32_v1/fwd/ground_truth.jsonl.
    python results/cc4d_k32_reg.py"""
import collections, gzip, json, pathlib, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).parent
OUT = H.parent / "docs/cc4d/new_experiment/figs"
rows = json.load(gzip.open(H / "cc4d_k32_v1_gemini4_argon_20261109/rows.json.gz", "rt"))
cols = ["o", "k", "rep", "f", "a", "ok", "ab", "nv", "end", "n", "nm"]
D = {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}
gt = {}
for l in open(H.parent / "data/cc4d_k32_v1/fwd/ground_truth.jsonl"):
    g = json.loads(l); gt[g["report_id"], g["field_id"]] = g["target"]
reps = np.unique(D["rep"]); ridx = np.searchsorted(reps, D["rep"])
KS = ["8", "16", "32", "all"]
norm = D["ab"] == 0
rng = np.random.default_rng(0)
draws = rng.integers(0, len(reps), (500, len(reps)))
cell = np.array([re.sub(r"^step\d+_", "", f) for f in D["f"]])  # correct/error_types/duration/<recording fields>
y = D["ok"].astype(float)

# ---------- 7.7 regression ----------
key = np.array([r + "|" + f for r, f in zip(D["rep"], D["f"])])
_, qi = np.unique(key, return_inverse=True)
X_all = {
    "log2 batch size (call length)": np.log2(D["n"].astype(float)),
    "log2 (questions after it + 1)": np.log2(D["end"] + 1.0),
    "reversed (1 = yes)": (D["o"] == "rev").astype(float),
}


def fit(mask, X, w):
    """within-question demeaned weighted least squares; w = per-recording weight"""
    wr = w[ridx] * mask
    sw = np.bincount(qi, wr)
    dm = lambda v: v - (np.bincount(qi, wr * v) / np.maximum(sw, 1e-9))[qi]
    Z = np.column_stack([dm(v) for v in X]); yy = dm(y)
    A = np.einsum("ni,n,nj->ij", Z, wr, Z); return np.linalg.lstsq(A, np.einsum("ni,n,n->i", Z, wr, yy), rcond=None)[0]


def coef(mask, names, label):
    X = [X_all[n] for n in names]; m = mask & norm
    b0 = fit(m, X, np.ones(len(reps)))
    bs = np.array([fit(m, X, np.bincount(d, minlength=len(reps)).astype(float)) for d in draws])
    print(f"{label:16}", "  ".join(f"{n.split(' (')[0][:22]:22} {100*b0[i]:+.2f} [{100*np.percentile(bs[:, i],2.5):+.2f}, {100*np.percentile(bs[:, i],97.5):+.2f}]" for i, n in enumerate(names)))
    return b0, bs


print("=== 7.7 effect on accuracy (pt) per doubling, question fixed effects, 95% bootstrap over recordings ===")
N3 = list(X_all)
print("-- both regressors, all questions")
coef(np.ones(len(y), bool), N3, "all")
print("-- batch size only / position only (what the earlier pooled slope mixed)")
coef(np.ones(len(y), bool), N3[:1] + N3[2:], "batch size only")
coef(np.ones(len(y), bool), N3[1:], "position only")
print("-- by content")
for c in ("correct", "error_types", "duration"):
    coef(cell == c, N3, c)
coef(~np.isin(cell, ["correct", "error_types", "duration"]), N3, "recording-level")
print("-- by answer type")
for a in ("bool", "multi_choice", "seconds"):
    coef(D["a"] == a, N3, a)

# ---------- 7.8 baselines ----------
# base = best constant answer within the cell, on the normal questions (same rule as cc4d_q2.py)
tol = lambda t: max(2.0, 0.25 * t)
normal_keys = {}
for r, f_, a_ in zip(D["rep"], D["f"], D["a"]):
    if gt[r, f_] != "not_visible": normal_keys[r, f_] = a_
cells = collections.defaultdict(list)
for (r, f_), a_ in normal_keys.items():
    cells[re.sub(r"^step\d+_", "", f_)].append(gt[r, f_])


def baseline(name, ts):
    a = {"correct": "bool", "order_violated": "bool", "error_types": "multi", "steps_done": "int", "longest_step": "single"}.get(name, "seconds")
    if a == "bool": return max(np.mean([t is True for t in ts]), np.mean([t is False for t in ts]))
    if a == "multi": return collections.Counter(json.dumps(sorted(t)) for t in ts).most_common(1)[0][1] / len(ts)
    if a == "int": return collections.Counter(ts).most_common(1)[0][1] / len(ts)
    if a == "single":
        return max(1 / 8, collections.Counter(ts if isinstance(ts[0], str) else [x[0] for x in ts]).most_common(1)[0][1] / len(ts))
    grid = np.linspace(0, max(ts) + 60, 800)  # seconds: best constant guess, an upper bound
    return float(max(np.mean([abs(g - t) <= tol(t) for t in ts]) for g in grid))


BASE = {c: baseline(c, ts) for c, ts in cells.items()}
print("\n=== 7.8 baselines (best constant answer) ===")
for c in ("correct", "error_types", "duration", "order_violated", "steps_done", "total_duration", "longest_step"):
    print(f"  {c:15} n={len(cells[c]):4}  base {BASE[c]:.3f}")
bq = np.array([BASE[c] for c in cell])  # per-question baseline


def nacc(mask):
    """pooled normalized accuracy: (acc - mean base) / (1 - mean base), bootstrap over recordings"""
    m = mask & norm
    s = lambda v: np.bincount(ridx[m], v[m], len(reps))
    a, b, n = s(y), s(bq), np.bincount(ridx[m], minlength=len(reps))
    f = lambda w: ((w @ a / (w @ n)) - (w @ b / (w @ n))) / (1 - (w @ b / (w @ n)))
    bs = [f(np.bincount(d, minlength=len(reps)).astype(float)) for d in draws]
    return f(np.ones(len(reps))), *np.percentile(bs, [2.5, 97.5])


fmt = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"
print("\n-- pooled normalized accuracy by k (0 = best constant answer, 1 = all correct)")
for o in ("fwd", "rev"):
    print(o, "  ".join(f"k={k}: {fmt(nacc((D['o']==o)&(D['k']==k)))}" for k in KS))
print("both", "  ".join(f"k={k}: {fmt(nacc(D['k']==k))}" for k in KS))
print("-- by cell (both orders)")
NC = {}
for c in ("correct", "error_types", "duration", "order_violated", "steps_done", "total_duration", "longest_step"):
    m = cell == c
    NC[c] = [nacc(m & (D["k"] == k)) for k in KS]
    raw = [y[m & norm & (D["k"] == k)].mean() for k in KS]
    print(f"  {c:15} base {BASE[c]:.2f}  raw " + " ".join(f"{100*r:.0f}" for r in raw) + "  norm " + "  ".join(f"{t[0]:+.2f}" for t in NC[c]))
# error_types by mistakes: a constant 'no mistake' answer gets 0 on steps with mistakes
print("-- error_types: steps with mistakes (constant 'no mistake' baseline = 0)")
for lab, v in (("no mistake", 1), ("has mistakes", 0)):
    print(f"  {lab:13}", " ".join(f"k={k}:{100*y[norm & (D['nm']==v) & (D['k']==k)].mean():.1f}" for k in KS))

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
lab = {"correct": "done correctly? (bool)", "error_types": "what mistakes? (multi)", "duration": "step seconds", "order_violated": "order violated (bool)", "steps_done": "steps done (int)", "total_duration": "total seconds", "longest_step": "longest step (single)"}
for c in ("correct", "error_types", "duration"):
    ax[0].plot(range(4), [t[0] for t in NC[c]], marker="o", label=lab[c])
ax[0].axhline(0, color="k", lw=.8, ls="--"); ax[0].set_xticks(range(4), KS); ax[0].set_xlabel("questions per call (k)"); ax[0].set_ylabel("normalized accuracy"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[0].set_title("Step questions, baseline-normalized"); ax[0].set_ylim(-0.1, 0.8)
names = ["batch size", "distance from end"]
b0, bs = coef(np.ones(len(y), bool), N3[:2], "all (2 regressors)")
for i, nm in enumerate(names):
    ax[1].errorbar([i], [100 * b0[i]], [[100 * (b0[i] - np.percentile(bs[:, i], 2.5))], [100 * (np.percentile(bs[:, i], 97.5) - b0[i])]], marker="o", capsize=4)
ax[1].axhline(0, color="k", lw=.8, ls="--"); ax[1].set_xticks(range(2), ["per doubling of\nbatch size", "per doubling of distance\nfrom the end"]); ax[1].set_xlim(-.6, 1.6)
ax[1].set_ylabel("accuracy change (pt)"); ax[1].grid(alpha=.3); ax[1].set_title("Batch size and position, separated")
plt.tight_layout(); plt.savefig(OUT / "q7_separate.png", dpi=130); plt.close()
