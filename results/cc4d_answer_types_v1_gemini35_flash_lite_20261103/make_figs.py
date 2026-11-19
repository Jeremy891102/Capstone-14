"""Rebuild figs/*.png from runs/*/evaluations/*/field_scores.jsonl (offline)."""
import collections, glob, json, os
import merged
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{here}/figs", exist_ok=True)


load = merged.load


def acc(run, row, t):
    rr = [r for r in load(run) if r["field_id"][:3] == row and r["extra"]["answer_type"] == t
          and not r["extra"].get("not_done_item")]
    return sum(r["correct"] is True for r in rr) / len(rr)


def hal(run):
    rr = [r for r in load(run) if r["extra"].get("not_done_item")]
    return sum(r["extra"].get("hallucination") is True for r in rr) / len(rr)


cells = [("err", t) for t in ["bool", "single_choice", "multi_choice", "int", "open_text"]] + \
        [("dur", t) for t in ["bool", "single_choice", "seconds"]] + \
        [("loc", t) for t in ["bool", "single_choice", "seconds"]]

TC = {t: f"C{i}" for i, t in enumerate(["bool", "single_choice", "multi_choice", "int", "open_text", "seconds"])}

# Fig 1: Q1 question count, main experiment
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for a, row in zip(ax, ["err", "dur", "loc"]):
    for c in [c for c in cells if c[0] == row]:
        a.plot([1, 4, 8], [acc(f"fl_base_k{k}", *c) for k in (1, 4, 8)], marker="o", label=c[1], color=TC[c[1]])
    a.set_title(row); a.set_xticks([1, 4, 8]); a.set_xlabel("questions per call")
h = {}
for a in ax:
    h.update(zip(*a.get_legend_handles_labels()[::-1]))
fig.legend(h.values(), h.keys(), loc="center right", fontsize=7)
ax[0].set_ylabel("accuracy")
plt.tight_layout(rect=(0, 0, .88, 1)); plt.savefig(f"{here}/figs/q1_count.png", dpi=130); plt.close()

# Fig 2: Q1 E5 up to 16
fig, ax = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
for a, row in zip(ax, ["loc", "dur"]):
    for t in ["bool", "single_choice", "seconds"]:
        a.plot([1, 4, 8, 16], [acc(f"fl_e5_{row}_base_k{k}", row, t) for k in (1, 4, 8, 16)], marker="o", label=t)
    a.set_title(f"E5 {row}"); a.set_xscale("log", base=2); a.set_xticks([1, 4, 8, 16]); a.set_xticklabels([1, 4, 8, 16])
    a.set_xlabel("questions per call")
ax[0].set_ylabel("accuracy"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{here}/figs/q1_e5.png", dpi=130); plt.close()

# Fig 3: Q2/Q3 type x content at k=1
fig, ax = plt.subplots(figsize=(8, 3.4))
labels = [f"{r}\n{t.replace('_choice','')}" for r, t in cells]
ax.bar(labels, [acc("fl_base_k1", *c) for c in cells], color=[{"err": "C3", "dur": "C0", "loc": "C2"}[c[0]] for c in cells])
ax.set_ylabel("accuracy (k=1)"); ax.tick_params(axis="x", labelsize=7)
plt.tight_layout(); plt.savefig(f"{here}/figs/q2_type_content.png", dpi=130); plt.close()

# Fig 4: Q5 hallucination by count
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.plot([1, 4, 8], [hal(r) for r in ("fl_not_done_k1", "fl_h4_k4", "fl_h8_k8")], marker="o", label="main")
for row in ("loc", "dur"):
    ax.plot([1, 4, 8, 16], [hal(f"fl_e5_{row}_{r}") for r in ("not_done_k1", "h4_k4", "h8_k8", "h16_k16")], marker="o", label=f"E5 {row}")
ax.set_xlabel("questions per call"); ax.set_ylabel("hallucination rate"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{here}/figs/q5_hallucination.png", dpi=130); plt.close()

# Fig 5: Q6 forward vs reversed
fig, ax = plt.subplots(figsize=(5.5, 3.4))
def tot(run):
    rr = [r for r in load(run) if not r["extra"].get("not_done_item")]
    return sum(r["correct"] is True for r in rr) / len(rr)
ks = [4, 8]
ax.bar([i - .2 for i in range(2)], [tot(f"fl_base_k{k}") for k in ks], .4, label="forward")
ax.bar([i + .2 for i in range(2)], [tot(f"fl_base_rev_k{k}") for k in ks], .4, label="reversed")
ax.set_xticks(range(2)); ax.set_xticklabels([f"k={k}" for k in ks]); ax.set_ylabel("accuracy"); ax.set_ylim(.3, .45); ax.legend()
plt.tight_layout(); plt.savefig(f"{here}/figs/q6_order.png", dpi=130); plt.close()
