"""Q4: does mixing answer types in one call change the answers to the same questions? E4 (mixed, k = 8) vs E3 (single type, k = 8).
Only the 4 target questions of each E4 call are scored, paired with the same (report, field) in E3. In E4 the targets always sit
at positions 1-4 (forward) or 5-8 (reversed), while E3 puts F1-F4 at 1-4 and F5-F8 at 5-8, so half of the pairs also change
position. The position-matched pairs are the clean test of mixing; the others mix it with a position change.
Two models (Flash-Lite, Qwen); Argon has no E4."""
import json, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from cc4d_q1_logit import gm, qw, here
from cc4d_q1_logit2 import wald

rng = np.random.default_rng(0)
B = 2000
OUT = f"{here}/../docs/cc4d/result/figs"
DATA = f"{here}/../data/cc4d_answer_types_v1"
SHORT = {"single_choice": "single", "multi_choice": "multi"}
COL = {"Flash-Lite": "#e07b39", "Qwen": "#3b6fb6"}
LOAD = {"Flash-Lite": gm.load, "Qwen": qw.load_all}


def index_pos(group):
    out = {}
    for l in open(f"{DATA}/{group}/reports.jsonl"):
        r = json.loads(l)
        for i, f in enumerate(r["fields"]):
            out[r["id"], f["id"]] = i
    return out


POS = {g: index_pos(g) for g in ("e4", "e4_rev", "base", "base_rev")}


def pairs(model):
    rows = []
    for order, e4g, e3g in (("fwd", "e4", "base"), ("rev", "e4_rev", "base_rev")):
        e3 = {(r["report_id"], r["field_id"]): r for r in LOAD[model](f"fl_{e3g}_k8")}
        for r in LOAD[model](f"fl_{e4g}_k8"):
            rep = r["report_id"]; base = re.sub(r"_half\d$", "", rep); grp = base.split("__")[1]
            if r["extra"].get("not_done_item") or not grp.endswith(r["field_id"].split("_", 1)[1]) or r["correct"] is None:
                continue
            a = e3.get((base, r["field_id"]))
            if a is None or a["correct"] is None:
                continue
            p4, p3 = POS[e4g][rep, r["field_id"]], POS[e3g][base, r["field_id"]]
            rows.append(dict(model=model, order=order, rec=re.match(r"cc4d-(\d+_\d+)__", rep).group(1), fact=(base, r["field_id"]),
                             cell=f"{r['field_id'][:3]} {SHORT.get(r['extra']['answer_type'], r['extra']['answer_type'])}",
                             y3=int(a["correct"] is True), y4=int(r["correct"] is True),
                             mv="same" if p4 == p3 else ("earlier" if p4 < p3 else "later")))
    return pd.DataFrame(rows)


d = pd.concat([pairs(m) for m in LOAD])
d["delta"] = d.y4 - d.y3
recs = sorted(d.rec.unique()); RI = {r: i for i, r in enumerate(recs)}
draws = rng.integers(0, len(recs), (B, len(recs)))


def boot(x):
    """mean of (mixed - single) in percentage points and 95% interval, bootstrap over recordings"""
    s, n = np.zeros(len(recs)), np.zeros(len(recs))
    for r, v in zip(x.rec, x.delta):
        s[RI[r]] += v; n[RI[r]] += 1
    smp = s[draws].sum(1) / np.maximum(n[draws].sum(1), 1)
    return 100 * s.sum() / n.sum(), 100 * np.percentile(smp, [2.5, 97.5])


def fmt(x):
    v, (lo, hi) = boot(x)
    return f"{v:+.1f} [{lo:+.1f}, {hi:+.1f}] (n={len(x)}, right->wrong {int(((x.y3 == 1) & (x.y4 == 0)).sum())}, wrong->right {int(((x.y3 == 0) & (x.y4 == 1)).sum())})"


print("shift groups (E4 position vs E3 position):", d.groupby(["order", "mv"]).size().to_dict())

print("\n=== 4.1 does mixing change accuracy? position-matched pairs only (clean test), pt [95% CI] ===")
for m in LOAD:
    for o in ("fwd", "rev"):
        print(f"{m:10s} {o}: pooled {fmt(d[(d.model == m) & (d.order == o) & (d.mv == 'same')])}")
    print(f"{m:10s} fwd+rev: pooled {fmt(d[(d.model == m) & (d.mv == 'same')])}")
print("\nper cell (position-matched, forward + reversed pooled):")
for m in LOAD:
    print(m)
    for c in sorted(d.cell.unique()):
        print(f"  {c:14s} {fmt(d[(d.model == m) & (d.cell == c) & (d.mv == 'same')])}")

print("\n=== 4.2 does the mixing effect depend on the target type / content? (logit on stacked E3 + E4, position-matched pairs) ===")
for m in LOAD:
    x = d[(d.model == m) & (d.mv == "same")]
    st = pd.concat([x.assign(mixed=0, y=x.y3), x.assign(mixed=1, y=x.y4)])
    st["rid"] = pd.factorize(st.rec)[0]
    r = smf.logit("y ~ mixed * C(cell)", st).fit(disp=0, cov_type="cluster", cov_kwds={"groups": st.rid})
    print(m); wald(r, st, r"^mixed:", "mixed x cell (does the effect differ by cell?)")
    r0 = smf.logit("y ~ mixed + C(cell)", st).fit(disp=0, cov_type="cluster", cov_kwds={"groups": st.rid})
    lo, hi = r0.conf_int().loc["mixed"]; print(f"  pooled mixed coef {r0.params['mixed']:+.3f} [{lo:+.3f}, {hi:+.3f}] p={r0.pvalues['mixed']:.3f}")

print("\n=== 4.3 mixing or position? mixed - single, by whether the target moved to an earlier / later / same position ===")
for m in LOAD:
    for o in ("fwd", "rev"):
        for s in ("same", "earlier", "later"):
            x = d[(d.model == m) & (d.order == o) & (d.mv == s)]
            if len(x):
                print(f"{m:10s} {o} {s:8s} {fmt(x)}")

print("\n=== 4.4 do the models agree? (position-matched; logit y ~ mixed * model + cell) ===")
x = d[d.mv == "same"]
st = pd.concat([x.assign(mixed=0, y=x.y3), x.assign(mixed=1, y=x.y4)]); st["rid"] = pd.factorize(st.rec)[0]
r = smf.logit("y ~ mixed * C(model) + C(cell)", st).fit(disp=0, cov_type="cluster", cov_kwds={"groups": st.rid})
wald(r, st, r"^mixed:", "mixed x model")

# ---------- figures ----------
lab = {}
cells = sorted(d.cell.unique())
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for a, m in zip(ax, LOAD):
    for j, o in enumerate(("fwd", "rev")):
        for i, c in enumerate(cells):
            x = d[(d.model == m) & (d.order == o) & (d.mv == "same") & (d.cell == c)]
            v, (lo, hi) = boot(x)
            a.errorbar(i + (j - .5) * .25, v, yerr=[[v - lo], [hi - v]], fmt="o" if o == "fwd" else "s", color=COL[m], mfc=COL[m] if o == "fwd" else "white", capsize=2, label=o if (o, id(a)) not in lab and not lab.update({(o, id(a)): 1}) else None)
    a.axhline(0, color="k", ls="--", lw=1); a.set_xticks(range(len(cells))); a.set_xticklabels(cells, rotation=60, fontsize=7); a.set_title(m); a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("mixed − single type (pt)"); ax[0].set_ylim(-25, 25); ax[0].legend(fontsize=7, title="order")
plt.tight_layout(); plt.savefig(f"{OUT}/q4_mixed_cells.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for a, m in zip(ax, LOAD):
    for i, s in enumerate(("earlier", "same", "later")):
        for j, o in enumerate(("fwd", "rev")):
            x = d[(d.model == m) & (d.order == o) & (d.mv == s)]
            if len(x):
                v, (lo, hi) = boot(x)
                a.errorbar(i + (j - .5) * .25, v, yerr=[[v - lo], [hi - v]], fmt="o" if o == "fwd" else "s", color=COL[m], mfc=COL[m] if o == "fwd" else "white", capsize=3, label=o if (o, id(a)) not in lab and not lab.update({(o, id(a)): 1}) else None)
    a.axhline(0, color="k", ls="--", lw=1); a.set_xticks(range(3)); a.set_xticklabels(["target moved earlier", "same position", "target moved later"], fontsize=8); a.set_title(m); a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("mixed − single type (pt)"); ax[0].set_ylim(-25, 25); ax[0].legend(fontsize=7, title="order")
plt.tight_layout(); plt.savefig(f"{OUT}/q4_mixed_position.png", dpi=130); plt.close()
