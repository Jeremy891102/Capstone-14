"""Figures for docs/cc4d/result/q1.md. Accuracy by question count; 95% interval = bootstrap over recordings (2,000)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cc4d_q1_logit2 import table

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../docs/cc4d/result/figs")
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
COL = {"Flash-Lite": "#e07b39", "Qwen": "#3b6fb6", "Argon": "#3a9a5b"}
KS = [1, 4, 8]
d = table(["base_k1", "base_k4", "base_k8"])
d = d[d.groupby(["model", "fact"]).k.transform("nunique") == 3]
common = ["dur bool", "dur single", "loc bool", "loc single", "loc seconds"]


def curve(x, col="y"):
    """mean and bootstrap-by-recording interval per k"""
    g = x.groupby(["rec", "k"])[col].agg(["sum", "count"]).unstack("k").fillna(0)
    s, n = g["sum"][KS].values, g["count"][KS].values
    b = rng.integers(0, len(g), (2000, len(g)))
    boot = np.array([s[i].sum(0) / n[i].sum(0) for i in b])
    return 100 * s.sum(0) / n.sum(0), 100 * np.percentile(boot, [2.5, 97.5], axis=0)


def panel(ax, x, title):
    for m, c in COL.items():
        y = x[x.model == m]
        if not len(y):
            continue
        mu, ci = curve(y)
        ax.errorbar(range(3), mu, yerr=[mu - ci[0], ci[1] - mu], marker="o", color=c, label=m, capsize=3)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"k={k}" for k in KS]); ax.set_title(title, fontsize=10)


def grid(parts, name, ylabel="accuracy (%)"):
    fig, ax = plt.subplots(1, len(parts), figsize=(3.4 * len(parts), 3.3), squeeze=False, sharey=True)
    for a, (t, x) in zip(ax[0], parts):
        panel(a, x, t)
    ax[0][0].set_ylabel(ylabel); ax[0][0].set_ylim(0, 100); ax[0][-1].legend(fontsize=7)
    for a in ax[0]:
        a.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(f"{out}/{name}", dpi=130); plt.close()


c5 = d[d.cell.isin(common)]
grid([("5 common cells, pooled", c5)], "q1_overall.png")
grid([(f"{c}  (bool + single)", d[d.type.isin(["bool", "single"]) & (d.content == c)]) for c in ("err", "dur", "loc")], "q1_content.png")
grid([(t, d[(d.type == t) & (d.content != "err")]) for t in ("bool", "single", "seconds")], "q1_type_durloc.png")
grid([(t, d[(d.type == t) & (d.content == "err")]) for t in ("bool", "single", "multi", "int", "open_text")], "q1_type_err.png")

# bias on bool: share of "yes" answers by the true answer
b = d[(d.type == "bool") & d.pred.isin([True, False])].copy()
b["yes"] = b.pred.astype(int)
fig, ax = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True); ax[0].set_ylim(0, 100)
for a, m in zip(ax, COL):
    for tgt, ls in ((True, "-"), (False, "--")):
        x = b[(b.model == m) & (b.target == tgt)]
        a.plot(range(3), [100 * x[x.k == k].yes.mean() for k in KS], ls, marker="o", color=COL[m],
               label=f"true answer = {'yes' if tgt else 'no'}")
    a.set_xticks(range(3)); a.set_xticklabels([f"k={k}" for k in KS]); a.set_title(m, fontsize=10); a.legend(fontsize=7)
ax[0].set_ylabel('share answered "yes" (%)')
plt.tight_layout(); plt.savefig(f"{out}/q1_bool_bias.png", dpi=130); plt.close()
