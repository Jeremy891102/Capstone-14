"""Abstention with vs without video, per model / order / k (docs/cc4d/new_experiment/novideo.zh.md).
  hallucination  = skipped-step questions answered with anything but not_visible / all skipped-step questions
  false abstain  = filmed-step questions answered not_visible / all filmed-step questions   (field starts with "step")
Run from the repo root: python results/cc4d_k32_abstention.py > results/cc4d_k32_abstention_out.txt"""
import gzip, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "results/cc4d_k32_v1_"
P = {"Qwen3-VL-32B": (D + "qwen3vl32b_20261110", D + "qwen3vl32b_novideo_20261111"),
     "Argon": (D + "gemini4_argon_20261109", D + "gemini4_argon_novideo_20261111")}

def rates(rows, o, k):
    sk = [r for r in rows if r[0] == o and r[1] == k and r[6] == 1]
    fl = [r for r in rows if r[0] == o and r[1] == k and r[6] == 0 and r[3].startswith("step")]
    return 100 * (1 - sum(r[7] for r in sk) / len(sk)), 100 * sum(r[7] for r in fl) / len(fl), len(sk), len(fl)

R = {}
print("model | order | k | halluc video | halluc no-video | false-abstain video | false-abstain no-video   (%; n skipped / n filmed)")
for m, (v, n) in P.items():
    V = json.load(gzip.open(v + "/rows.json.gz", "rt")); N = json.load(gzip.open(n + "/rows.json.gz", "rt"))
    for o in ("fwd", "rev"):
        for k in ("8", "16", "32", "all"):
            hv, fv, ns, nf = rates(V, o, k); hn, fn, _, _ = rates(N, o, k)
            R[m, o, k] = (hv, hn, fv, fn)
            print(f"{m} | {o} | {k} | {hv:5.1f} | {hn:5.1f} | {fv:5.1f} | {fn:5.1f}   (n {ns} / {nf})")

# top = hallucination, bottom = false abstention; left Qwen, right Argon; blue = with video, grey = no video, solid = fwd, dashed = rev
K = ("8", "16", "32", "all")
fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharey="row")
for j, m in enumerate(P):
    for i, (name, off) in enumerate((("skipped-step hallucination (%)", 0), ("false abstention on filmed steps (%)", 2))):
        a = ax[i, j]
        for o, ls in (("fwd", "-"), ("rev", "--")):
            for vid, c, lab in ((0, "#1f77b4", "with video"), (1, "#9a9a9a", "no video")):
                a.plot(K, [R[m, o, k][off + vid] for k in K], ls, marker="o", color=c, label=f"{lab}, {o}")
        a.set_ylim(0, 105) if i == 0 else a.set_ylim(0, 25)
        a.grid(alpha=0.3); a.set_xlabel("k (questions per call)")
        if j == 0: a.set_ylabel(name)
        if i == 0: a.set_title(m)
        if i == 0 and j == 0: a.legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig("docs/cc4d/new_experiment/figs/abstention.png", dpi=130)
