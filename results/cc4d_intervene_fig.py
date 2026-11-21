"""Figure for docs/attention/attention_intervention*.md.

python results/cc4d_intervene_fig.py SCORES.json OUT.png     (SCORES = results/cc4d_intervene_score.py output)
x: mean late-layer video share of the generated tokens' attention; y: accuracy minus base, 95% interval
over reports. The band is vLLM k = 8 minus base (the accuracy a k = 8 prompt would give).
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8d8c86"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 11,
    }
)
S = json.load(open(sys.argv[1]))["summary"]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
k8 = S["vllm_k8"]["diff"]
ax.axhspan(k8[1], k8[2], color=GRAY, alpha=0.18, lw=0)
ax.axhline(k8[0], color=GRAY, lw=1.2, ls="--")
ax.text(36.5, k8[0] + 0.25, f"vLLM k = 8: {k8[0]:+.1f} pp (band = 95% interval)", color=MUTED, fontsize=9.5)
ax.axhline(0, color=INK, lw=1)
names = {"base": "base", "a1.5": "α 1.5", "a2": "α 2", "a3": "α 3", "all2": "α 2, all layers", "text2": "text ×2 (control)"}
for n, label in names.items():
    x = S[n]["late"]
    d = S[n]["diff"] or [0.0, 0.0, 0.0]
    color = ORANGE if n == "text2" else BLUE
    ax.errorbar(x, d[0], yerr=[[d[0] - d[1]], [d[2] - d[0]]], fmt="o", color=color, capsize=3, ms=7, lw=1.5)
    y, off, ha = {"a2": (d[1], (6, -16), "left"), "all2": (d[1], (-6, -16), "right")}.get(n, (d[2], (0, 7), "center"))
    ax.annotate(label, (x, y), textcoords="offset points", xytext=off, ha=ha, fontsize=9.5, color=INK)
ax.set_xlabel("Late-layer video share of the answer's attention (%)")
ax.set_ylabel("Accuracy minus base (pp)")
ax.set_xlim(33, 73)
ax.set_ylim(-2.2, 5.2)
ax.set_title("More attention on the video, same accuracy (k = all, 54 reports)", loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=160)
