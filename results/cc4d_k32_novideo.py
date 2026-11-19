"""No-video control (docs/cc4d/new_experiment/novideo.zh.md): accuracy with vs without video, by k and answer type.
Rows format: see cc4d_k32.py. Run from the repo root: python results/cc4d_k32_novideo.py > results/cc4d_k32_novideo_out.txt"""
import gzip, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "results/cc4d_k32_v1_"
P = {"Qwen3-VL-32B": (D + "qwen3vl32b_20261110/rows.json.gz", D + "qwen3vl32b_novideo_20261111/rows.json.gz"),
     "Argon": (D + "gemini4_argon_20261109/rows.json.gz", D + "gemini4_argon_novideo_20261111/rows.json.gz")}
K=["8","16","32","all"]; TYPES=["bool","multi_choice","seconds","int","single_choice"]
rng = np.random.default_rng(0)
def load(p):
    R=json.load(gzip.open(p,"rt")); recs=sorted({r[2] for r in R}); ix={r:i for i,r in enumerate(recs)}
    return R,ix
def table(R,ix,sel):  # per (k, recording): [sum, n]
    n=len(ix); a={k:np.zeros((n,2)) for k in K}
    for r in R:
        if sel(r): a[r[1]][ix[r[2]]]+= (r[5] if True else 0,1)
    return a
out={}
RES, BOOT = {}, {}
for m,(pv,pn) in P.items():
    V,ix=load(pv); N,ix2=load(pn); assert ix==ix2
    B=rng.integers(0,len(ix),(1000,len(ix))); BOOT[m]=B
    def acc(a,k,b=None):
        x=a[k] if b is None else a[k][b]; return 100*x[...,0].sum(-1)/x[...,1].sum(-1)
    def ci(v): return np.percentile(v,[2.5,97.5])
    for name,sel in [("normal",lambda r:r[6]==0)]+[(t,(lambda r,t=t:r[6]==0 and r[4]==t)) for t in TYPES]:
        av,an=table(V,ix,sel),table(N,ix,sel)
        print(f"== {m} {name}")
        RES[m,name]=(av,an)
        for k in K:
            g=acc(av,k)-acc(an,k); gb=acc(av,k,B)-acc(an,k,B)
            print(f"  k={k:3} video {acc(av,k):5.1f} novideo {acc(an,k):5.1f} gain {g:5.1f} [{ci(gb)[0]:5.1f},{ci(gb)[1]:5.1f}]")
        d=(acc(av,"all")-acc(an,"all"))-(acc(av,"8")-acc(an,"8")); db=(acc(av,"all",B)-acc(an,"all",B))-(acc(av,"8",B)-acc(an,"8",B))
        print(f"  gain(all)-gain(8) {d:5.1f} [{ci(db)[0]:5.1f},{ci(db)[1]:5.1f}]")
    # hallucination on abstention Qs: 1 - said_not_visible
    for lab,R_ in (("video",V),("novideo",N)):
        row=[]
        for k in K:
            x=[r[7] for r in R_ if r[1]==k and r[6]==1]; row.append(100-100*sum(x)/len(x))
        print(f"  hallucination {lab}:",[round(v,1) for v in row])

# ---- figure: top = accuracy by k (normal questions), bottom = video gain by answer type, k=8 vs k=all
COL = {"video": "#1f77b4", "novideo": "#9a9a9a"}
fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharey="row")
def acc(a, k, b=None):
    x = a[k] if b is None else a[k][b]; return 100 * x[..., 0].sum(-1) / x[..., 1].sum(-1)
for j, m in enumerate(P):
    av, an = RES[m, "normal"]; B = BOOT[m]; a = ax[0, j]
    for lab, arr in (("with video", av), ("no video", an)):
        y = np.array([acc(arr, k) for k in K]); lo, hi = np.percentile(np.array([acc(arr, k, B) for k in K]), [2.5, 97.5], axis=1)
        c = COL["video" if lab == "with video" else "novideo"]
        a.plot(K, y, "o-", color=c, label=lab); a.fill_between(K, lo, hi, color=c, alpha=0.15)
    a.set_title(m); a.set_ylim(20, 70); a.set_xlabel("k (questions per call)"); a.grid(alpha=0.3)
    if j == 0: a.set_ylabel("accuracy (%)"); a.legend(loc="upper right")
    b = ax[1, j]; w = 0.38
    for i, (k, c) in enumerate((("8", "#1f77b4"), ("all", "#d62728"))):
        g = [acc(RES[m, t][0], k) - acc(RES[m, t][1], k) for t in TYPES]
        b.bar(np.arange(len(TYPES)) + (i - 0.5) * w, g, w, color=c, label=f"k = {k}")
    b.axhline(0, color="k", lw=0.8); b.set_xticks(range(len(TYPES))); b.set_xticklabels(TYPES, fontsize=8); b.grid(axis="y", alpha=0.3)
    if j == 0: b.set_ylabel("video gain (pt)"); b.legend(loc="upper left")
fig.suptitle("Left: Qwen3-VL-32B, right: Argon. Top: accuracy by k. Bottom: with-video minus no-video, by answer type", fontsize=9)
fig.tight_layout(); fig.savefig("docs/cc4d/new_experiment/figs/novideo.png", dpi=130)
