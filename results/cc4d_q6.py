"""Q6: does the order / position of a question inside the call affect accuracy? Every k > 1 batch is asked forward and reversed
(a fact at position p forward sits at k + 1 - p reversed). Three models: Flash-Lite, Qwen (k = 4, 8; E5 k = 4, 8, 16) and Argon
(base k = 8 only, 5 cells). Filmed facts only; paired by (report, field).
6.1 rev - fwd accuracy; 6.2 position inside the call (same fact, late minus early); 6.3 by cell and by k; 6.4 invalid answers;
6.5 models."""
import json, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from cc4d_q1_logit import LOAD, SHORT, here
from cc4d_q1_logit2 import wald

rng = np.random.default_rng(0)
B = 2000
OUT = f"{here}/../docs/cc4d/result/figs"
DATA = f"{here}/../data/cc4d_answer_types_v1"
COL = {"Flash-Lite": "#e07b39", "Qwen": "#3b6fb6", "Argon": "#3a9a5b"}
_idx = {}


def idx(g):
    if g not in _idx:
        for l in open(f"{DATA}/{g}/reports.jsonl"):
            r = json.loads(l)
            for i, f in enumerate(r["fields"]):
                _idx[g, r["id"], f["id"]] = i
    return g


def rows(model, group, k):
    """group: base / e5_loc_base / e5_dur_base ; returns forward and reversed rows for one k"""
    out = []
    for o, tag in (("fwd", ""), ("rev", "_rev")):
        name = group.replace("_base", f"_base{tag}") if group != "base" else f"base{tag}"
        run, dg = f"{name}_k{k}", idx(name)
        try:
            rr = LOAD[model](run)
        except Exception:
            return None
        for r in rr:
            if r["extra"].get("not_done_item") or r["correct"] is None:
                continue
            c = r["extra"]["answer_type"]
            out.append(dict(model=model, exp=group, k=k, order=o, rev=int(o == "rev"), rep=r["report_id"], field=r["field_id"],
                            rec=re.match(r"cc4d-(\d+_\d+)__", r["report_id"]).group(1),
                            cell=f"{r['field_id'][:3]} {SHORT.get(c, c)}", y=int(r["correct"] is True),
                            invalid=int(r["outcome"] in ("invalid_answer", "invalid_output", "missing_answer")),
                            pos=_idx[dg, r["report_id"], r["field_id"]] % k + 1))
    return out


R = []
for m in LOAD:
    for g, ks in (("base", (4, 8)), ("e5_loc_base", (4, 8, 16)), ("e5_dur_base", (4, 8, 16))):
        for k in ks:
            x = rows(m, g, k)
            if x:
                R += x
d = pd.DataFrame(R)
d["posfrac"] = (d.pos - 1) / (d.k - 1)
d["main"] = d.exp == "base"
# keep facts present in both orders
d = d[d.groupby(["model", "exp", "k", "rep", "field"]).order.transform("nunique") == 2]
recs = sorted(d.rec.unique()); RI = {r: i for i, r in enumerate(recs)}
draws = rng.integers(0, len(recs), (B, len(recs)))
print("pairs (facts in both orders):", d.groupby(["model", "exp", "k"]).size().div(2).astype(int).to_string())


def paired(x):
    """one row per fact: forward and reversed outcome"""
    w = x.pivot_table(index=["rep", "field", "rec", "cell"], columns="order", values=["y", "pos", "invalid"], aggfunc="first").reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    return w


def boot(vals, rec_col):
    s, n = np.zeros(len(recs)), np.zeros(len(recs))
    for r, v in zip(rec_col, vals):
        s[RI[r]] += v; n[RI[r]] += 1
    smp = s[draws].sum(1) / np.maximum(n[draws].sum(1), 1)
    return 100 * s.sum() / n.sum(), 100 * np.percentile(smp, [2.5, 97.5])


def fmt(vals, rec_col):
    v, (lo, hi) = boot(vals, rec_col)
    return f"{v:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]"


# ---------- 6.1 rev - fwd ----------
print("\n=== 6.1 reversed - forward accuracy (pt) [95% CI], same fact ===")
P = {}
for (m, e, k), x in d.groupby(["model", "exp", "k"]):
    w = paired(x); P[m, e, k] = w
    print(f"{m:10s} {e:12s} k={k:2d} fwd {100 * w.y_fwd.mean():5.1f}% rev {100 * w.y_rev.mean():5.1f}%  rev-fwd {fmt(w.y_rev - w.y_fwd, w.rec)} (n={len(w)})")

# ---------- 6.2 position ----------
print("\n=== 6.2 position: same fact, late position minus early position (pt); facts at the middle (odd k) excluded ===")
LE = {}
for (m, e, k), w in P.items():
    w = w.copy(); w["late_minus_early"] = np.where(w.pos_fwd > w.pos_rev, w.y_fwd - w.y_rev, w.y_rev - w.y_fwd)
    w = w[w.pos_fwd != w.pos_rev]; LE[m, e, k] = w
    print(f"{m:10s} {e:12s} k={k:2d} late - early {fmt(w.late_minus_early, w.rec)}")
print("\naccuracy by position, k = 8 main, forward / reversed:")
for m in LOAD:
    x = d[(d.model == m) & (d.exp == "base") & (d.k == 8)]
    if len(x):
        print(f"  {m:10s} " + " ".join(f"p{p} {100 * x[(x.pos == p) & (x.order == 'fwd')].y.mean():.0f}/{100 * x[(x.pos == p) & (x.order == 'rev')].y.mean():.0f}" for p in range(1, 9)))
print("\nlogit y ~ posfrac + rev + cell (k = 8, main): posfrac = position effect, rev = reversed-order effect at the same position")
for m in LOAD:
    x = d[(d.model == m) & (d.exp == "base") & (d.k == 8)]
    if not len(x):
        continue
    r = smf.logit("y ~ posfrac + rev + C(cell)", x).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(x.rec)[0]})
    ci = r.conf_int()
    print(f"  {m:10s} posfrac {r.params['posfrac']:+.3f} [{ci.loc['posfrac', 0]:+.3f}, {ci.loc['posfrac', 1]:+.3f}] p={r.pvalues['posfrac']:.3f} | rev {r.params['rev']:+.3f} [{ci.loc['rev', 0]:+.3f}, {ci.loc['rev', 1]:+.3f}] p={r.pvalues['rev']:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for a, m in zip(ax, LOAD):
    x = d[(d.model == m) & (d.exp == "base") & (d.k == 8)]
    for o, mk, mfc in (("fwd", "o", None), ("rev", "s", "white")):
        a.plot(range(1, 9), [100 * x[(x.pos == p) & (x.order == o)].y.mean() for p in range(1, 9)], mk + "-", color=COL[m], mfc=mfc or COL[m], label=o)
    a.set_title(m); a.set_xlabel("position of the question in the call"); a.set_ylim(0, 100); a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("accuracy (%)"); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/q6_by_position.png", dpi=130); plt.close()

# ---------- 6.3 by cell and by k ----------
print("\n=== 6.3 rev - fwd by cell, k = 8 (pt) ===")
cells = sorted(d.cell.unique())
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for a, m in zip(ax, LOAD):
    w = P.get((m, "base", 8))
    if w is None:
        continue
    print(m)
    for i, c in enumerate(cells):
        z = w[w.cell == c]
        if not len(z):
            continue
        v, (lo, hi) = boot(z.y_rev - z.y_fwd, z.rec)
        print(f"   {c:14s} {v:+5.1f} [{lo:+5.1f}, {hi:+5.1f}] (n={len(z)})")
        a.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="o", color=COL[m], capsize=2)
    a.axhline(0, color="k", ls="--", lw=1); a.set_xticks(range(len(cells))); a.set_xticklabels(cells, rotation=60, fontsize=7); a.set_title(m); a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("reversed − forward (pt)"); ax[0].set_ylim(-30, 15)
plt.tight_layout(); plt.savefig(f"{OUT}/q6_by_cell.png", dpi=130); plt.close()
for m in LOAD:
    x = d[(d.model == m) & (d.exp == "base") & (d.k == 8)]
    if len(x):
        r = smf.logit("y ~ rev * C(cell)", x).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(x.rec)[0]})
        wald(r, x, r"^rev:", f"{m}: rev x cell (k = 8)")
print("\nrev - fwd by k (main for k = 4, 8; E5 loc / dur for k = 4, 8, 16):")
for m in LOAD:
    for e in ("base", "e5_loc_base", "e5_dur_base"):
        line = " | ".join(f"k={k} {fmt(P[m, e, k].y_rev - P[m, e, k].y_fwd, P[m, e, k].rec)}" for k in (4, 8, 16) if (m, e, k) in P)
        if line:
            print(f"  {m:10s} {e:12s} {line}")
    x = d[(d.model == m) & (d.exp == "base")]
    if x.k.nunique() > 1:
        r = smf.logit("y ~ rev * np.log2(k) + C(cell)", x).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(x.rec)[0]})
        print(f"    {m}: rev x log2(k) coef {r.params['rev:np.log2(k)']:+.3f} p={r.pvalues['rev:np.log2(k)']:.3f}")

# ---------- 6.4 invalid answers ----------
print("\n=== 6.4 invalid answers (counted as wrong): share, forward vs reversed; and rev - fwd among pairs valid in both ===")
for (m, e, k), w in P.items():
    if k < 8 and e == "base":
        continue
    both = w[(w.invalid_fwd == 0) & (w.invalid_rev == 0)]
    print(f"{m:10s} {e:12s} k={k:2d} invalid fwd {100 * w.invalid_fwd.mean():4.1f}% rev {100 * w.invalid_rev.mean():4.1f}% | valid in both (n={len(both)}): rev-fwd {fmt(both.y_rev - both.y_fwd, both.rec)}")

# ---------- 6.5 models ----------
print("\n=== 6.5 do the models agree? 5 common cells, k = 8 (logit y ~ rev * model + cell) ===")
common = ["dur bool", "dur single", "loc bool", "loc single", "loc seconds"]
x = d[(d.exp == "base") & (d.k == 8) & d.cell.isin(common)]
for m in LOAD:
    w = P[m, "base", 8]; w = w[w.cell.isin(common)]
    print(f"  {m:10s} rev-fwd {fmt(w.y_rev - w.y_fwd, w.rec)} (n={len(w)})")
r = smf.logit("y ~ rev * C(model, Treatment('Argon')) + C(cell)", x).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(x.rec)[0]})
wald(r, x, r"^rev:", "rev x model")
