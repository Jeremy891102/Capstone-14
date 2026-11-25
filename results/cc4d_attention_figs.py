"""Figures for docs/attention/attention_probe.zh.md.

python results/cc4d_attention_figs.py ATTN_DIR OUT_DIR ROWS.json.gz
ATTN_DIR holds k32_{fwd,rev}_kall{,_B}.jsonl and k32_fwd_k8.jsonl (scripts/qwen_attention.py output).
Writes attn_position.png, attn_k8_vs_kall.png, attn_layers.png.
"""

import collections
import gzip
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# probe measure: "attention" (value tokens) or "attention_pred" (predicting positions)
KEY = os.environ.get("ATTN_KEY", "attention")

attn_dir, out_dir, rows_path = sys.argv[1:4]
rng = np.random.default_rng(0)

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8d8c86"  # palette slots 1-3 + neutral
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 11,
        "lines.linewidth": 2,
    }
)


def load(name):
    rows = []
    for line in open(f"{attn_dir}/{name}.jsonl"):
        a = json.loads(line)
        rows.append(
            {**{k: a[k] for k in ("report_id", "field_id", "index")}, "x": np.array(a[KEY])}
        )
    return rows


# ---- figure 1: where the answer tokens' attention goes, by position (k = all, A + B, fwd + rev)
rows = [
    r for n in ("k32_fwd_kall", "k32_rev_kall", "k32_fwd_kall_B", "k32_rev_kall_B") for r in load(n)
]
parts = {  # name -> column of the 4-way split, mean over layers
    "earlier answers": 3,
    "attention sink (before the video)": 1,
    "question text": 2,
    "video": 0,
}
val = {k: np.array([r["x"][:, c].mean() * 100 for r in rows]) for k, c in parts.items()}
qid = collections.defaultdict(list)
for i, r in enumerate(rows):
    qid[r["report_id"], r["field_id"]].append(i)
idx = np.array([r["index"] for r in rows])
bins = [(0, 4), (5, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 99)]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
colors = {
    "earlier answers": ORANGE,
    "attention sink (before the video)": GRAY,
    "question text": AQUA,
    "video": BLUE,
}
for name, v in val.items():
    adj = v.copy()  # within-question: subtract the question's own mean, add the overall mean
    for ix in qid.values():
        adj[ix] = v[ix] - v[ix].mean() + v.mean()
    xs = [idx[(idx >= lo) & (idx <= hi)].mean() for lo, hi in bins]
    ys = [adj[(idx >= lo) & (idx <= hi)].mean() for lo, hi in bins]
    ax.plot(
        xs,
        ys,
        color=colors[name],
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    ax.text(xs[-1] + 1.2, ys[-1], name, color=INK, va="center", fontsize=10.5)
ax.set_xlim(0, 80)
ax.set_xlabel("position of the question in the prompt (index)")
ax.set_ylabel("share of the answer's attention (%)")
ax.set_title(
    "Qwen3-VL-32B, k = all: the video's share stays flat while earlier answers take over",
    loc="left",
    fontsize=11.5,
)
fig.text(
    0.01,
    0.01,
    "54 reports, 5,678 answers; mean over layers and heads; same question's own mean removed "
    "(forward and reversed)",
    color=MUTED,
    fontsize=8.5,
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{out_dir}/attn_position.png", dpi=170)

# ---- figures 2, 3: k = 8 vs k = all, same question, same position (first 8 of the report), class A
k8 = {(r["report_id"], r["field_id"]): r for r in load("k32_fwd_k8")}
al = {(r["report_id"], r["field_id"]): r for r in load("k32_fwd_kall")}
first = [q for q in k8 if q in al and al[q]["index"] == k8[q]["index"] < 8]
kind = {  # (report, field) -> answer type, from the k = all rows of cc4d_k32_rows.py
    (r[2], r[3]): r[4]
    for r in json.load(gzip.open(rows_path, "rt"))
    if r[0] == "fwd" and r[1] == "all"
}


def late(x):
    m = x[len(x) * 3 // 4 :]
    return (m[:, 0] / (m[:, 0] + m[:, 2])).mean() * 100


def per_layer(x):
    return x[:, 0] / (x[:, 0] + x[:, 2]) * 100


reps = sorted({q[0] for q in first})
by = collections.defaultdict(list)
for q in first:
    by[q[0]].append(q)


def ci(f, qs_of):
    est = np.mean([f(q) for r in reps for q in qs_of(r)])
    bs = [
        np.mean([f(q) for r in rng.choice(reps, len(reps)) for q in qs_of(r)]) for _ in range(500)
    ]
    return est, *np.percentile(bs, [2.5, 97.5])


groups = [
    ("all questions", lambda q: True),
    ("bool\n(done correctly?)", lambda q: kind[q] == "bool"),
    ("multi-choice\n(what mistakes?)", lambda q: kind[q] == "multi_choice"),
    ("seconds\n(step duration)", lambda q: kind[q] == "seconds"),
]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
w = 0.36
ticks = []
for j, (lab, sel) in enumerate(groups):
    qs_of = lambda r, sel=sel: [q for q in by[r] if sel(q)]  # noqa: E731
    n = sum(len(qs_of(r)) for r in reps)
    for off, d, col, nm in ((-w / 2, k8, BLUE, "k = 8"), (w / 2, al, ORANGE, "k = all")):
        est, lo, hi = ci(lambda q, d=d: late(d[q]["x"]), qs_of)
        xpos = j + off
        ax.bar(xpos, est, width=w, color=col, label=nm if j == 0 else None)
        ax.errorbar(xpos, est, yerr=[[est - lo], [hi - est]], color=INK, capsize=3, linewidth=1.2)
        ax.text(xpos, hi + 1.2, f"{est:.0f}", ha="center", color=INK, fontsize=10)
    ticks.append(f"{lab}\nn = {n}")
ax.set_xticks(range(len(groups)), ticks)
ax.set_ylim(0, 80)
ax.set_ylabel("video share of attention, last quarter of layers (%)")
ax.set_title(
    "Same question, same position: the video gets less late-layer attention at k = all",
    loc="left",
    fontsize=11.5,
)
ax.legend(frameon=False, loc="upper right", ncol=2)
fig.text(
    0.01,
    0.01,
    "Class A, 23 reports; first 8 questions of each report; "
    "video / (video + question text); 95% interval over reports",
    color=MUTED,
    fontsize=8.5,
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{out_dir}/attn_k8_vs_kall.png", dpi=170)

fig, ax = plt.subplots(figsize=(8.2, 4.4))
L = len(first and k8[first[0]]["x"])
for d, col, nm in ((k8, BLUE, "k = 8"), (al, ORANGE, "k = all")):
    prof = np.mean([per_layer(d[q]["x"]) for q in first], 0)
    ax.plot(range(L), prof, color=col, label=nm)

ax.axvspan(L * 3 // 4, L - 1, color=GRID, alpha=0.6, linewidth=0)
ax.text(L * 3 // 4 + 0.5, 3, "last quarter of layers", color=MUTED, fontsize=9)
ax.set_xlim(0, L - 1)
ax.legend(frameon=False, loc="upper left")
ax.set_xlabel("text layer")
ax.set_ylabel("video share: video / (video + question text) (%)")
ax.set_title(
    "The k = 8 vs k = all difference sits in the last quarter of layers", loc="left", fontsize=11.5
)
fig.text(
    0.01,
    0.01,
    "Class A, 23 reports; same 184 questions at the same position; mean over heads",
    color=MUTED,
    fontsize=8.5,
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{out_dir}/attn_layers.png", dpi=170)
print("wrote 3 figures to", out_dir)
