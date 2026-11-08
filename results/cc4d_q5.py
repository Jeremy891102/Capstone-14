"""Q5: hallucination. A not-done step (recipe step with no footage) has the target `not_visible`; any other valid answer is a
hallucination, `not_visible` on a filmed step is a false not-done. Two models (Flash-Lite, Qwen); Argon was not run on these groups.
5.1 by answer type (k = 1); 5.2 by question count (main k = 1/4/8, E5 up to 16) and the trade-off with false not-done;
5.3 by the not-done step's position in the call; 5.4 does a not-done step in the call change the other questions (H batch vs
original batch); 5.5 false alarms in the error row (no-mistake step called wrong).
Hallucination rate = hallucinated / valid answers on not-done steps (invalid answers are left out)."""
import json, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from cc4d_q1_logit import gm, qw, here

rng = np.random.default_rng(0)
B = 2000
OUT = f"{here}/../docs/cc4d/result/figs"
DATA = f"{here}/../data/cc4d_answer_types_v1"
SHORT = {"single_choice": "single", "multi_choice": "multi"}
COL = {"Flash-Lite": "#e07b39", "Qwen": "#3b6fb6"}
LOAD = {"Flash-Lite": gm.load, "Qwen": qw.load_all}
_idx, _gt = {}, {}


def meta(g):
    """field index in its report, ground-truth step id"""
    if g not in _idx:
        for l in open(f"{DATA}/{g}/reports.jsonl"):
            r = json.loads(l)
            for i, f in enumerate(r["fields"]):
                _idx[g, r["id"], f["id"]] = i
        for l in open(f"{DATA}/{g}/ground_truth.jsonl"):
            r = json.loads(l); _gt[g, r["report_id"], r["field_id"]] = (r["evidence"]["step_id"], r["evidence"].get("role"), r["target"])
    return g


def rec_of(rep):
    return re.match(r"cc4d-(\d+_\d+)__", rep).group(1)


def rows_for(model, run, dgroup, k, kind, order="fwd", exp="main"):
    """kind: 'nd' not-done items, 'filmed' other items"""
    meta(dgroup); out = []
    for r in LOAD[model](run):
        nd = bool(r["extra"].get("not_done_item"))
        if (kind == "nd") != nd:
            continue
        rep, f = r["report_id"], r["field_id"]
        step, role, tgt = _gt[dgroup, rep, f]
        out.append(dict(model=model, exp=exp, k=k, order=order, rec=rec_of(rep), rep=rep, field=f, step=step, tgt=tgt,
                        cell=f"{f[:3] if not nd else re.match(r'(err|dur|loc)', f).group(1)} {SHORT.get(r['extra']['answer_type'], r['extra']['answer_type'])}",
                        pos=_idx[dgroup, rep, f] % k + 1, outcome=r["outcome"], y=int(r["correct"] is True),
                        pred=r["predicted"], fnd=int(r["extra"].get("false_not_done") is True)))
    return out


# ---------- load not-done items for every k ----------
ND = []
for m in LOAD:
    ND += rows_for(m, "fl_not_done_k1", "not_done", 1, "nd")
    for k in (4, 8):
        for o, tag in (("fwd", ""), ("rev", "_rev")):
            ND += rows_for(m, f"fl_h{k}{tag}_k{k}", f"h{k}{tag}", k, "nd", o)
    for c in ("loc", "dur"):
        ND += rows_for(m, f"fl_e5_{c}_not_done_k1", f"e5_{c}_not_done", 1, "nd", "fwd", "e5_" + c)
        for k in (4, 8, 16):
            for o, tag in (("fwd", ""), ("rev", "_rev")):
                ND += rows_for(m, f"fl_e5_{c}_h{k}{tag}_k{k}", f"e5_{c}_h{k}{tag}", k, "nd", o, "e5_" + c)
nd = pd.DataFrame(ND)
nd["valid"] = nd.outcome.isin(["correct", "incorrect"]); nd["hall"] = (nd.outcome == "incorrect").astype(int)
nd = nd[nd.valid]
nd["type"] = nd.cell.str.split().str[1]
recs = sorted(nd.rec.unique()); RI = {r: i for i, r in enumerate(recs)}
draws = rng.integers(0, len(recs), (B, len(recs)))


def boot(x, col="hall", scale=100):
    s, n = np.zeros(len(recs)), np.zeros(len(recs))
    for r, v in zip(x.rec, x[col]):
        s[RI[r]] += v; n[RI[r]] += 1
    if n.sum() == 0:
        return np.nan, np.array([np.nan, np.nan])
    smp = s[draws].sum(1) / np.maximum(n[draws].sum(1), 1)
    return scale * s.sum() / n.sum(), scale * np.percentile(smp, [2.5, 97.5])


def f(x, col="hall"):
    v, (lo, hi) = boot(x, col)
    return f"{v:5.1f}% [{lo:4.1f}, {hi:4.1f}] (n={len(x)})"


print("not-done items loaded:", nd.groupby(["model", "exp", "k", "order"]).size().unstack("order").to_string())

# ---------- 5.1 by answer type, k = 1 ----------
print("\n=== 5.1 hallucination rate at k = 1 (E1, main), by answer type; % of valid answers on not-done steps ===")
e1 = nd[(nd.exp == "main") & (nd.k == 1)]
for m in LOAD:
    print(m, "overall", f(e1[e1.model == m]))
    for c in sorted(e1.cell.unique()):
        print(f"   {c:14s} {f(e1[(e1.model == m) & (e1.cell == c)])}")
fl_e1 = {m: pd.DataFrame(rows_for(m, "fl_base_k1", "base", 1, "filmed")) for m in LOAD}
print("\nfalse not-done at k = 1 (filmed steps answered not_visible):")
for m in LOAD:
    x = fl_e1[m]; print(m, "overall", f"{100 * x.fnd.mean():.1f}%", " | ".join(f"{c} {100 * g.fnd.mean():.1f}%" for c, g in x.groupby("cell")))

fig, ax = plt.subplots(figsize=(10, 3.6))
cells = sorted(e1.cell.unique())
for j, m in enumerate(LOAD):
    for i, c in enumerate(cells):
        v, (lo, hi) = boot(e1[(e1.model == m) & (e1.cell == c)])
        ax.errorbar(i + (j - .5) * .25, v, yerr=[[v - lo], [hi - v]], fmt="o", color=COL[m], capsize=2, label=m if i == 0 else None)
ax.set_xticks(range(len(cells))); ax.set_xticklabels(cells, rotation=45, fontsize=8); ax.set_ylim(0, 100); ax.set_ylabel("hallucination rate (%)"); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/q5_by_type.png", dpi=130); plt.close()

# ---------- 5.2 by question count ----------
print("\n=== 5.2 hallucination rate by question count (same not-done steps at every k; forward order) ===")
main = nd[(nd.exp == "main") & (nd.order == "fwd")].copy()
main["key"] = main.rec + "|" + main.cell + "|" + main.step.astype(str)
common = main.groupby(["model", "key"]).k.transform("nunique") == 3
for m in LOAD:
    x = main[(main.model == m) & common]
    print(m, " | ".join(f"k={k} {f(x[x.k == k])}" for k in (1, 4, 8)))
    r = smf.logit("hall ~ np.log2(k) + C(cell)", x).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(x.rec)[0]})
    lo, hi = r.conf_int().loc["np.log2(k)"]; print(f"   logit per doubling of k: {r.params['np.log2(k)']:+.3f} [{lo:+.3f}, {hi:+.3f}] p={r.pvalues['np.log2(k)']:.3f}")
print("\nby type, hallucination at k = 1 / 4 / 8 (common items):")
for m in LOAD:
    x = main[(main.model == m) & common]
    for t in ["bool", "single", "multi", "int", "open_text", "seconds"]:
        g = x[x.type == t]
        print(f"   {m:10s} {t:10s} " + " / ".join(f"{100 * g[g.k == k].hall.mean():4.0f}%" for k in (1, 4, 8)))
print("\nE5 (longer videos), k = 1 / 4 / 8 / 16, forward:")
for m in LOAD:
    for c in ("loc", "dur"):
        x = nd[(nd.exp == "e5_" + c) & (nd.order == "fwd") & (nd.model == m)]
        print(f"  {m:10s} {c} " + " | ".join(f"k={k} {100 * x[x.k == k].hall.mean():4.0f}% (n={int((x.k == k).sum())})" for k in (1, 4, 8, 16)))
print("\nfalse not-done (filmed steps answered not_visible) by k, base group, forward:")
FN = {}
for m in LOAD:
    FN[m] = pd.concat([pd.DataFrame(rows_for(m, f"fl_base_k{k}", "base", k, "filmed")) for k in (1, 4, 8)])
    print(f"  {m:10s} " + " | ".join(f"k={k} {100 * FN[m][FN[m].k == k].fnd.mean():.1f}%" for k in (1, 4, 8)))
    print("     by type: " + "; ".join(f"{t} " + "/".join(f"{100 * g[g.k == k].fnd.mean():.0f}" for k in (1, 4, 8)) for t, g in FN[m].assign(t=FN[m].cell.str.split().str[1]).groupby("t")))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for a, (ttl, getter) in zip(ax, (("hallucination (not-done steps)", lambda m, k: main[common & (main.model == m) & (main.k == k)].hall.mean()),
                                  ("false not-done (filmed steps)", lambda m, k: FN[m][FN[m].k == k].fnd.mean()))):
    for m in LOAD:
        a.plot(range(3), [100 * getter(m, k) for k in (1, 4, 8)], "o-", color=COL[m], label=m)
    a.set_xticks(range(3)); a.set_xticklabels(["k=1", "k=4", "k=8"]); a.set_title(ttl, fontsize=10); a.set_ylim(0, 100); a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("rate (%)"); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/q5_by_k.png", dpi=130); plt.close()

# ---------- 5.3 position of the not-done step inside the call ----------
print("\n=== 5.3 hallucination by the not-done step's position in the call (H batches, forward + reversed) ===")
h = nd[(nd.k > 1) & (nd.exp == "main")].copy(); h["posfrac"] = (h.pos - 1) / (h.k - 1)
for m in LOAD:
    x = h[h.model == m]
    r = smf.logit("hall ~ posfrac + np.log2(k) + C(cell)", x).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(x.rec)[0]})
    lo, hi = r.conf_int().loc["posfrac"]
    print(f"{m}: posfrac coef {r.params['posfrac']:+.3f} [{lo:+.3f}, {hi:+.3f}] p={r.pvalues['posfrac']:.3f}  (0 = first, 1 = last)")
    x8 = x[x.k == 8]
    print("   k=8 by position 1..8: " + " ".join(f"{100 * x8[x8.pos == p].hall.mean():.0f}%(n={int((x8.pos == p).sum())})" for p in range(1, 9)))
    x4 = x[x.k == 4]
    print("   k=4 by position 1..4: " + " ".join(f"{100 * x4[x4.pos == p].hall.mean():.0f}%(n={int((x4.pos == p).sum())})" for p in range(1, 5)))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for a, k in zip(ax, (4, 8)):
    for m in LOAD:
        x = h[(h.model == m) & (h.k == k)]
        v = [100 * x[x.pos == p].hall.mean() for p in range(1, k + 1)]
        a.plot(range(1, k + 1), v, "o-", color=COL[m], label=m)
    a.set_title(f"k = {k}"); a.set_xlabel("position of the not-done step"); a.set_ylim(0, 100); a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("hallucination rate (%)"); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/q5_by_position.png", dpi=130); plt.close()

# ---------- 5.4 does a not-done step change the other questions? ----------
print("\n=== 5.4 other questions of an H batch (H batch) vs the same question in the original batch; accuracy pt [95% CI] ===")
for m in LOAD:
    for k in (4, 8):
        for o, tag in (("fwd", ""), ("rev", "_rev")):
            hb = pd.DataFrame(rows_for(m, f"fl_h{k}{tag}_k{k}", f"h{k}{tag}", k, "filmed", o))
            bs = pd.DataFrame(rows_for(m, f"fl_base{tag}_k{k}", f"base{tag}", k, "filmed", o))
            j = hb.merge(bs, on=["rep", "field"], suffixes=("_h", "_b"))
            j["rec"] = j.rec_h; j["d"] = j.y_h - j.y_b
            v, (lo, hi) = boot(j, "d")
            print(f"  {m:10s} k={k} {o}: {v:+.1f} [{lo:+.1f}, {hi:+.1f}] (n={len(j)}, right->wrong {int(((j.y_b == 1) & (j.y_h == 0)).sum())}, wrong->right {int(((j.y_b == 0) & (j.y_h == 1)).sum())})")

# ---------- 5.5 false alarms in the error row ----------
print("\n=== 5.5 error row: no-mistake step called wrong (1 - accuracy on no-mistake steps), by type and k ===")
for m in LOAD:
    rows = []
    for k in (1, 4, 8):
        x = pd.DataFrame(rows_for(m, f"fl_base_k{k}", "base", k, "filmed"))
        x = x[x.cell.str.startswith("err")]
        x["t"] = x.cell.str.split().str[1]
        rows.append(x)
    x = pd.concat(rows)
    # a fact has a mistake if its err bool target is False
    mist = {(r.rec, r.field.split("_")[0]): r.tgt is False for r in x[(x.k == 1) & (x.t == "bool")].itertuples()}
    x["nomist"] = [not mist[(r, f_.split("_")[0])] for r, f_ in zip(x.rec, x.field)]
    x = x[x.nomist]
    print(m + "  (false-alarm % at k = 1 / 4 / 8)")
    for t in ["bool", "single", "multi", "int", "open_text"]:
        g = x[x.t == t]
        print(f"   {t:10s} " + " / ".join(f"{100 * (1 - g[g.k == k].y.mean()):4.0f}%" for k in (1, 4, 8)))
