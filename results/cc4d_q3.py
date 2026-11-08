"""Q3: how does each model do on each content (error / duration / localization)? E1 (k = 1), base group.
3.1 chance-normalized accuracy per content within bool and single (+ seconds: loc vs dur); 3.2 bool balanced accuracy
(is it real discrimination?); 3.3 do the models rank the contents the same way; 3.4 same at k = 4 and 8.
Reuses data and baselines of cc4d_q2.py; paired by recording (the same 64 recordings are in every content)."""
import contextlib, io
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import cc4d_q2 as q2
from cc4d_q1_logit import fit
from cc4d_q1_logit2 import table, wald

d, BASE, norm, per_rec, draws, ci, COL, OUT = q2.d, q2.BASE, q2.norm, q2.per_rec, q2.draws, q2.ci, q2.COL, q2.OUT
MODELS, CONTENTS = list(COL), ["err", "dur", "loc"]


def nacc(x, c):
    """normalized accuracy and bootstrap samples (shared draws, so differences are paired by recording)"""
    s, n = per_rec(x)
    return norm(s.sum() / n.sum(), c), norm(s[draws].sum(1) / np.maximum(n[draws].sum(1), 1), c)


def sub(m, c, t):
    return d[(d.model == m) & (d.content == c) & (d.type == t)]


def fmt(v, s):
    lo, hi = ci(s)
    return f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]"


# ---------- 3.1 ----------
print("=== 3.1 chance-normalized accuracy by content (k = 1) [95% CI]; differences paired by recording ===")
R = {}
for m in MODELS:
    for t in ("bool", "single", "seconds"):
        res = {c: nacc(sub(m, c, t), (c, t)) for c in CONTENTS if len(sub(m, c, t))}
        for c, (v, s) in res.items():
            R[m, t, c] = (v, ci(s))
        if len(res) < 2:
            continue
        print(f"{m:10s} {t:7s} " + "  ".join(f"{c} {fmt(*res[c])}" for c in res))
        pairs = [("dur", "err"), ("loc", "err"), ("loc", "dur")]
        print("           differences: " + "  ".join(f"{a}-{b} {fmt(res[a][0] - res[b][0], res[a][1] - res[b][1])}" for a, b in pairs if a in res and b in res))

lab = {}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for a, t in zip(ax, ("bool", "single", "seconds")):
    cs = CONTENTS if t != "seconds" else ["dur", "loc"]
    for j, m in enumerate(MODELS):
        for i, c in enumerate(cs):
            if (m, t, c) in R:
                v, (lo, hi) = R[m, t, c]
                a.errorbar(i + (j - 1) * 0.2, v, yerr=[[v - lo], [hi - v]], fmt="o", color=COL[m], capsize=2, label=m if not lab.get(m) and not lab.update({m: 1}) else None)
    a.axhline(0, color="k", ls="--", lw=1); a.set_xticks(range(len(cs))); a.set_xticklabels(cs); a.set_title(t); a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("chance-normalized accuracy\n(0 = baseline answer, 1 = all correct)"); ax[0].set_ylim(-0.6, 1.05); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/q3_content_norm.png", dpi=130); plt.close()

# ---------- 3.2 balanced accuracy of bool ----------
print("\n=== 3.2 bool balanced accuracy = mean of (right on true-yes, right on true-no); 0.5 = no discrimination ===")


def bal(x):
    sc = []
    for tgt in (True, False):
        s, n = per_rec(x[x.target == tgt])
        sc.append((s.sum() / n.sum(), s[draws].sum(1) / np.maximum(n[draws].sum(1), 1)))
    return (sc[0][0] + sc[1][0]) / 2, (sc[0][1] + sc[1][1]) / 2, sc[0][0], sc[1][0]


BAL = {}
for m in MODELS:
    out = {}
    for c in CONTENTS:
        x = sub(m, c, "bool")
        if len(x):
            out[c] = bal(x); BAL[m, c] = (out[c][0], ci(out[c][1]))
    print(f"{m:10s} " + "  ".join(f"{c} {out[c][0]:.3f} [{ci(out[c][1])[0]:.3f}, {ci(out[c][1])[1]:.3f}] (yes-right {out[c][2]:.2f} / no-right {out[c][3]:.2f})" for c in out))
    pairs = [("dur", "err"), ("loc", "err"), ("loc", "dur")]
    print("           differences: " + "  ".join(f"{a}-{b} {out[a][0] - out[b][0]:+.3f} [{ci(out[a][1] - out[b][1])[0]:+.3f}, {ci(out[a][1] - out[b][1])[1]:+.3f}]" for a, b in pairs if a in out and b in out))

lab = {}
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for j, m in enumerate(MODELS):
    for i, c in enumerate(CONTENTS):
        if (m, c) in BAL:
            v, (lo, hi) = BAL[m, c]
            ax.errorbar(i + (j - 1) * 0.2, v * 100, yerr=[[(v - lo) * 100], [(hi - v) * 100]], fmt="o", color=COL[m], capsize=2, label=m if not lab.get(m) and not lab.update({m: 1}) else None)
ax.axhline(50, color="k", ls="--", lw=1); ax.set_xticks(range(3)); ax.set_xticklabels(CONTENTS); ax.set_ylim(0, 100); ax.set_ylabel("bool balanced accuracy (%)")
ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/q3_bool_balanced.png", dpi=130); plt.close()

# ---------- 3.3 model x content ----------
print("\n=== 3.3 does the content ranking differ by model? logit y ~ model * content + type (bool + single, k = 1) ===")
M = "C(model, Treatment('Argon'))"
x = d[d.type.isin(["bool", "single"]) & (d.content != "err")].assign(k=1, lk=0)
r = fit(x, f"y ~ {M} * C(content, Treatment('dur')) + C(type)"); wald(r, x, r"C\(model.*:C\(content", "dur vs loc, three models")
x = d[d.type.isin(["bool", "single"]) & (d.model != "Argon")].assign(k=1, lk=0)
r = fit(x, "y ~ C(model) * C(content, Treatment('err')) + C(type)"); wald(r, x, r"C\(model.*:C\(content", "err/dur/loc, Flash-Lite vs Qwen")
x = d[(d.type == "seconds") & (d.content != "err") & (d.model != "Argon")].assign(k=1, lk=0)  # Argon has no dur seconds
r = fit(x, "y ~ C(model) * C(content, Treatment('dur'))"); wald(r, x, r"C\(model.*:C\(content", "seconds, dur vs loc, Flash-Lite vs Qwen")

# ---------- 3.4 the same at k = 4 and 8 ----------
print("\n=== 3.4 normalized accuracy by content at k = 1 / 4 / 8 (mean of bool and single) ===")
t = table(["base_k1", "base_k4", "base_k8"])
t = t[t.type.isin(["bool", "single"])]
t = t[t.groupby(["model", "fact"]).k.transform("nunique") == 3]
rows = []
for (m, c, k, ty), g in t.groupby(["model", "content", "k", "type"]):
    rows.append(dict(model=m, content=c, k=k, v=norm(g.y.mean(), (c, ty))))
print(pd.DataFrame(rows).groupby(["model", "content", "k"]).v.mean().unstack("k").round(2).to_string())
