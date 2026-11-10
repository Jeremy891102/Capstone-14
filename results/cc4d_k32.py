"""CC4D full-report batching (docs/cc4d/cc4d_k32_experiment_design.md), Gemini 4 Argon, 64 recordings.
Reads rows.json.gz (one row per question x condition) and prints the numbers used in docs/cc4d/new_experiment/q7.zh.md;
writes docs/cc4d/new_experiment/figs/q7_*.png.   python results/cc4d_k32.py
Row: order, k, report, field, answer_type, correct, is_abstention_q, said_not_visible, from_end, batch_len, no_mistake(-1 n/a)."""
import gzip, json, pathlib, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).parent
OUT = H.parent / "docs/cc4d/new_experiment/figs"
rows = json.load(gzip.open(H / "cc4d_k32_v1_gemini4_argon_20261109/rows.json.gz", "rt"))
cols = ["o", "k", "rep", "f", "a", "ok", "ab", "nv", "end", "n", "nm"]
D = {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}
reps = np.unique(D["rep"]); ridx = np.searchsorted(reps, D["rep"])
KS = ["8", "16", "32", "all"]
content = np.array([re.sub(r"^step\d+_", "", f) if re.match(r"step\d+_", f) else "recording" for f in D["f"]])
norm = D["ab"] == 0
rng = np.random.default_rng(0)
B = 1000
draws = rng.integers(0, len(reps), (B, len(reps)))  # shared bootstrap draws: all comparisons are paired by recording


def stat(mask, val):
    """mean of val over mask, with a 95% bootstrap interval over recordings"""
    s = np.bincount(ridx[mask], val[mask], len(reps)); n = np.bincount(ridx[mask], minlength=len(reps))
    b = s[draws].sum(1) / np.maximum(n[draws].sum(1), 1)
    return s.sum() / n.sum(), *np.percentile(b, [2.5, 97.5])


def f(x): return f"{100*x[0]:.1f} [{100*x[1]:.1f}, {100*x[2]:.1f}]"


ok, nv = D["ok"].astype(float), D["nv"].astype(float)
print("=== 1. accuracy on normal questions, by k and order (95% CI over recordings) ===")
acc = {}
for o in ("fwd", "rev"):
    for k in KS:
        acc[o, k] = stat(norm & (D["o"] == o) & (D["k"] == k), ok); print(o, k, f(acc[o, k]))
for k in KS: print("both orders", k, f(stat(norm & (D["k"] == k), ok)))
print("difference k=all minus k=8 (both orders):", end=" ")
s8 = lambda m: np.bincount(ridx[m], ok[m], len(reps)) / np.bincount(ridx[m], minlength=len(reps))
d = s8(norm & (D["k"] == "all")) - s8(norm & (D["k"] == "8")); b = d[draws].mean(1)
print(f"{100*d.mean():+.1f} pp [{100*np.percentile(b,2.5):+.1f}, {100*np.percentile(b,97.5):+.1f}]")

print("\n=== 2. by content / answer type (both orders) ===")
for name, key in (("content", content), ("answer type", D["a"])):
    for v in dict.fromkeys(key):
        print(f"{name:11} {v:13}", "  ".join(f"k={k}:{f(stat(norm & (key == v) & (D['k'] == k), ok))}" for k in KS))

print("\n=== 3. error_types split: steps with no mistake vs with mistakes (both orders) ===")
split = {}
for lab, v in (("no mistake", 1), ("has mistakes", 0)):
    for k in KS:
        split[lab, k] = stat(norm & (D["nm"] == v) & (D["k"] == k), ok)
    print(lab, "n =", int((norm & (D["nm"] == v) & (D["k"] == "all")).sum()) // 2, "per order;", "  ".join(f"k={k}:{f(split[lab,k])}" for k in KS))

print("\n=== 4. abstention: steps not done (hallucination = answered anything but not_visible); false abstention on normal questions ===")
hal, fa = {}, {}
for o in ("fwd", "rev"):
    for k in KS:
        m = (D["o"] == o) & (D["k"] == k)
        hal[o, k] = stat(m & (D["ab"] == 1), 1 - nv); fa[o, k] = stat(m & norm, nv)
        print(o, k, "hallucination", f(hal[o, k]), "| false abstention", f(fa[o, k]))
print("by answer type (both orders, hallucination):")
for a in ("bool", "multi_choice", "seconds"):
    print(" ", a, "  ".join(f"k={k}:{f(stat((D['ab']==1)&(D['a']==a)&(D['k']==k),1-nv))}" for k in KS))

print("\n=== 5. position: accuracy by questions-from-end; slope per doubling with question fixed effects ===")
edges = [(0, 4), (5, 12), (13, 24), (25, 40), (41, 99)]
pos = {}
for k in KS:
    for lo, hi in edges:
        m = norm & (D["k"] == k) & (D["end"] >= lo) & (D["end"] <= hi)
        if m.sum() > 200: pos[k, lo] = stat(m, ok)
key = np.array([r + "|" + f for r, f in zip(D["rep"], D["f"])]); _, qi = np.unique(key[norm], return_inverse=True)
x = np.log2(D["end"][norm] + 1.0); y = ok[norm]; ri = ridx[norm]


def slope(w):  # w: weight per recording (bootstrap multiplicity); within-question demeaned regression
    wr = w[ri]; sw = np.bincount(qi, wr); mx = np.bincount(qi, wr * x) / np.maximum(sw, 1e-9); my = np.bincount(qi, wr * y) / np.maximum(sw, 1e-9)
    dx, dy = x - mx[qi], y - my[qi]
    return (wr * dx * dy).sum() / (wr * dx * dx).sum()


def slope_ci(mask):
    m = mask[norm]; xx = np.where(m, x, 0)
    def one(w):
        wr = w[ri] * m; sw = np.bincount(qi, wr); mx = np.bincount(qi, wr * x) / np.maximum(sw, 1e-9); my = np.bincount(qi, wr * y) / np.maximum(sw, 1e-9)
        dx, dy = x - mx[qi], y - my[qi]; return (wr * dx * dy).sum() / (wr * dx * dx).sum()
    b = [one(np.bincount(dr, minlength=len(reps)).astype(float)) for dr in draws[:300]]
    return one(np.ones(len(reps))), *np.percentile(b, [2.5, 97.5])


for lab, m in (("all questions", np.ones(len(D["k"]), bool)),) + tuple((c, content == c) for c in ("correct", "error_types", "duration", "recording")):
    s = slope_ci(m); print(f"slope {lab:14} {100*s[0]:+.2f} pp [{100*s[1]:+.2f}, {100*s[2]:+.2f}] per doubling of distance from end")
print("(the slope pools all k; batch size and position are not separated here)")

# ---------- figures ----------
C = {"fwd": "#1f77b4", "rev": "#d95f02"}
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for o in ("fwd", "rev"):
    v = np.array([acc[o, k] for k in KS]) * 100
    ax[0].errorbar(range(4), v[:, 0], [v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], marker="o", capsize=3, label={"fwd": "forward", "rev": "reversed"}[o], color=C[o])
ax[0].set_xticks(range(4), ["8", "16", "32", "all"]); ax[0].set_xlabel("questions per call (k)"); ax[0].set_ylabel("accuracy, normal questions (%)")
ax[0].set_ylim(50, 70); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("Overall accuracy by batch size")
for v, lab in (("bool", "bool"), ("multi_choice", "multi-choice"), ("seconds", "seconds"), ("single_choice", "single-choice")):
    ax[1].plot(range(4), [100 * stat(norm & (D["a"] == v) & (D["k"] == k), ok)[0] for k in KS], marker="o", label=lab)
ax[1].set_xticks(range(4), ["8", "16", "32", "all"]); ax[1].set_xlabel("questions per call (k)"); ax[1].set_ylabel("accuracy (%)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].set_title("By answer type (both orders)")
plt.tight_layout(); plt.savefig(OUT / "q7_accuracy.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for o in ("fwd", "rev"):
    v = np.array([hal[o, k] for k in KS]) * 100
    ax[0].errorbar(range(4), v[:, 0], [v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], marker="o", capsize=3, label={"fwd": "forward", "rev": "reversed"}[o], color=C[o])
ax[0].set_xticks(range(4), ["8", "16", "32", "all"]); ax[0].set_xlabel("questions per call (k)"); ax[0].set_ylabel("answered something other than not_visible (%)")
ax[0].set_ylim(0, 100); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("Hallucination on steps not done")
for o in ("fwd", "rev"):
    ax[1].plot(range(4), [100 * fa[o, k][0] for k in KS], marker="o", color=C[o], label={"fwd": "forward", "rev": "reversed"}[o])
ax[1].set_xticks(range(4), ["8", "16", "32", "all"]); ax[1].set_xlabel("questions per call (k)"); ax[1].set_ylabel("answered not_visible (%)")
ax[1].set_ylim(0, 5); ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_title("False abstention on steps that were done")
plt.tight_layout(); plt.savefig(OUT / "q7_abstention.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for k, c in zip(KS, ["#999999", "#6baed6", "#2171b5", "#08306b"]):
    pts = [(lo, pos[k, lo]) for lo, _ in edges if (k, lo) in pos]
    ax[0].plot([i for i, (lo, _) in enumerate(edges) if (k, lo) in pos], [100 * p[0] for _, p in pts], marker="o", color=c, label=f"k = {k}")
ax[0].set_xticks(range(5), [f"{a}-{b}" if b < 99 else f"{a}+" for a, b in edges]); ax[0].set_xlabel("questions after it in the call")
ax[0].set_ylabel("accuracy (%)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_title("Accuracy by distance from the end of the call")
labs = ["no mistake", "has mistakes"]
for j, lab in enumerate(labs):
    ax[1].bar(np.arange(4) + (j - .5) * .38, [100 * split[lab, k][0] for k in KS], .38, label=lab)
ax[1].set_xticks(range(4), ["8", "16", "32", "all"]); ax[1].set_xlabel("questions per call (k)"); ax[1].set_ylabel("error_types accuracy (%)")
ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=.3); ax[1].set_ylim(0, 85); ax[1].set_title("'What mistakes?' accuracy")
plt.tight_layout(); plt.savefig(OUT / "q7_position_error.png", dpi=130); plt.close()
