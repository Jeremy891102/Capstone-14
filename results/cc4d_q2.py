"""Q2: the same fact asked in a different answer type (E1, k = 1, base group). Three models.
2.1 chance-normalized accuracy per type, paired difference from bool; 2.2 mistake / no-mistake split (error row) and
bool answer bias; 2.3 do the models rank the types the same way; 2.4 do answers to the same fact agree across types.
Baseline = max(chance, majority) as in the earlier Q2 analysis (seconds: best constant answer, an upper bound)."""
import collections, itertools, json, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cc4d_q1_logit import LOAD, fit, gm, here
from cc4d_q1_logit2 import wald

rng = np.random.default_rng(0)
B = 2000
OUT = f"{here}/../docs/cc4d/result/figs"
SHORT = {"single_choice": "single", "multi_choice": "multi"}
NOPT = {"err": 6, "dur": 4, "loc": 5}
TYPES = {"err": ["bool", "single", "multi", "int", "open_text"], "dur": ["bool", "single", "seconds"], "loc": ["bool", "single", "seconds"]}
COL = {"Flash-Lite": "#e07b39", "Qwen": "#3b6fb6", "Argon": "#3a9a5b"}

# ---------- data ----------
rows = []
for m, f in LOAD.items():
    for r in f("base_k1"):
        if r["extra"].get("not_done_item") or r["correct"] is None:
            continue
        rec = re.match(r"cc4d-(\d+_\d+)__", r["report_id"]).group(1)
        rows.append(dict(model=m, rec=rec, fact=r["field_id"].split("_")[0], content=r["field_id"][:3],
                         type=SHORT.get(r["extra"]["answer_type"], r["extra"]["answer_type"]), y=int(r["correct"] is True),
                         pred=r["predicted"], target=r["target"]))
d = pd.DataFrame(rows)
d["cell"] = d.content + " " + d.type
# a fact has a mistake if the err bool target is False (bool asks "done correctly?")
mist = {(r.rec, r.fact): (r.target is False) for r in d[(d.cell == "err bool") & (d.model == "Flash-Lite")].itertuples()}
d["mistake"] = [mist.get((r, f)) if c == "err" else None for r, f, c in zip(d.rec, d.fact, d.content)]

# ---------- baselines (same facts for every model: Flash-Lite file is complete) ----------
tol = {}
for run in gm.runs():
    for r in gm.load(run):
        if "tolerance" in r["extra"]:
            tol[r["report_id"].split("_half")[0], r["field_id"]] = r["extra"]["tolerance"]
fl = [r for r in gm.load("fl_base_k1") if not r["extra"].get("not_done_item")]
cellof = lambda r: (r["field_id"][:3], SHORT.get(r["extra"]["answer_type"], r["extra"]["answer_type"]))  # noqa: E731
asset = lambda t: set(t) if isinstance(t, list) else {t}  # noqa: E731


def baseline(c):
    rr = [r for r in fl if cellof(r) == c]; row, t = c
    if t == "bool":
        return max(np.mean([r["target"] is True for r in rr]), np.mean([r["target"] is False for r in rr]))
    if t == "single":
        letters = "ABCDEF"[: NOPT[row]]
        chance = float(np.mean([len(asset(r["target"])) / NOPT[row] for r in rr]))
        return max(chance, max(np.mean([L in asset(r["target"]) for r in rr]) for L in letters))
    if t in ("multi", "int"):
        key = (lambda r: json.dumps(sorted(r["target"]))) if t == "multi" else (lambda r: r["target"])
        return collections.Counter(key(r) for r in rr).most_common(1)[0][1] / len(rr)
    if t == "open_text":
        return float(np.mean([not r["target"] for r in rr]))
    med = float(np.median([v for (_, f), v in tol.items() if f.startswith(row)]))  # seconds
    tl = lambda r: tol.get((r["report_id"], r["field_id"])) or (max(2, 0.25 * r["target"]) if row == "dur" else med)  # noqa: E731
    pts = [(r["target"], tl(r)) for r in rr]
    grid = np.linspace(0, max(p for p, _ in pts) + 60, 600)
    return float(max(np.mean([abs(g - p) <= t_ for p, t_ in pts]) for g in grid))


BASE = {c: baseline(c) for c in sorted({cellof(r) for r in fl})}
norm = lambda a, c: (a - BASE[c]) / (1 - BASE[c])  # noqa: E731
recs = sorted(d.rec.unique()); RI = {r: i for i, r in enumerate(recs)}
draws = rng.integers(0, len(recs), (B, len(recs)))


def per_rec(x):
    """arrays of correct / count per recording"""
    n, s = np.zeros(len(recs)), np.zeros(len(recs))
    for r, y in zip(x.rec, x.y):
        n[RI[r]] += 1; s[RI[r]] += y
    return s, n


def acc_boot(x):
    s, n = per_rec(x)
    return s.sum() / n.sum(), s[draws].sum(1) / np.maximum(n[draws].sum(1), 1)


def ci(v):
    return np.percentile(v, [2.5, 97.5])


print("baselines (max of chance / majority):", {f"{a} {b}": round(v, 2) for (a, b), v in BASE.items()})

# ---------- 2.1 normalized accuracy and paired difference from bool ----------
print("\n=== 2.1 chance-normalized accuracy [95% CI]; difference from bool (paired by fact) ===")
N = {}
for m in LOAD:
    for c, ts in TYPES.items():
        x = d[(d.model == m) & (d.content == c)]
        if not len(x):
            continue
        for t in ts:
            xt = x[x.type == t]
            if not len(xt):
                continue
            a, s = acc_boot(xt); N[m, c, t] = (norm(a, (c, t)), ci(norm(s, (c, t))), a)
for m in LOAD:
    print(f"\n{m}")
    for c, ts in TYPES.items():
        for t in ts:
            if (m, c, t) not in N:
                continue
            v, (lo, hi), a = N[m, c, t]
            line = f"  {c} {t:10s} acc {a:.3f}  base {BASE[c, t]:.2f}  norm {v:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            if t != "bool":
                xb, xt = d[(d.model == m) & (d.cell == f"{c} bool")], d[(d.model == m) & (d.cell == f"{c} {t}")]
                sb, nb = per_rec(xb); st, nt = per_rec(xt)
                diff = norm(st.sum() / nt.sum(), (c, t)) - norm(sb.sum() / nb.sum(), (c, "bool"))
                bs = norm(st[draws].sum(1) / nt[draws].sum(1), (c, t)) - norm(sb[draws].sum(1) / nb[draws].sum(1), (c, "bool"))
                line += f"  | minus bool {diff:+.2f} [{ci(bs)[0]:+.2f}, {ci(bs)[1]:+.2f}]"
            print(line)

lab = {}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for a, c in zip(ax, TYPES):
    ts = TYPES[c]
    for j, m in enumerate(LOAD):
        for i, t in enumerate(ts):
            if (m, c, t) in N:
                v, (lo, hi), _ = N[m, c, t]
                a.errorbar(i + (j - 1) * 0.2, v, yerr=[[v - lo], [hi - v]], fmt="o", color=COL[m], capsize=2, label=m if not lab.get(m) and not lab.update({m: 1}) else None)
    a.axhline(0, color="k", ls="--", lw=1); a.set_xticks(range(len(ts))); a.set_xticklabels(ts, fontsize=8); a.set_title(c); a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("chance-normalized accuracy\n(0 = baseline answer, 1 = all correct)"); ax[0].set_ylim(-0.6, 1.05); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/q2_type_norm.png", dpi=130); plt.close()

# ---------- 2.2 ability vs bias ----------
print("\n=== 2.2 error row: accuracy on steps with a mistake / without (k = 1) ===")
e = d[d.content == "err"]
tab = e.groupby(["model", "type", "mistake"]).y.mean().unstack().round(3)
print(tab.to_string())
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for a, (flag, ttl) in zip(ax, ((True, "step has a mistake"), (False, "step has no mistake"))):
    for j, m in enumerate(("Flash-Lite", "Qwen")):
        a.bar(np.arange(5) + (j - .5) * 0.38, [tab.loc[(m, t), flag] * 100 for t in TYPES["err"]], 0.38, color=COL[m], label=m)
    a.axhline(50, color="k", ls=":", lw=1); a.set_xticks(range(5)); a.set_xticklabels(TYPES["err"], fontsize=8); a.set_title(ttl); a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("accuracy (%)"); ax[0].set_ylim(0, 100); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/q2_err_split.png", dpi=130); plt.close()
print("\n--- bool: share answered yes, by true answer (k = 1) ---")
b = d[(d.type == "bool") & d.pred.isin([True, False])].copy(); b["yes"] = b.pred.astype(int)
print(b.groupby(["model", "content", "target"]).yes.mean().unstack().round(3).to_string())

# ---------- 2.3 do models agree on how the types rank? ----------
print("\n=== 2.3 model x type interaction (logit y ~ model * type + cell-free content), cluster by recording ===")
M = "C(model, Treatment('Argon'))"; T = "C(type, Treatment('bool'))"
x = d[(d.content != "err") & d.type.isin(["bool", "single", "seconds"])].assign(k=1, lk=0)
r = fit(x, f"y ~ {M} * {T} + C(content)"); wald(r, x, r"C\(model.*:C\(type", "dur+loc: model x type")
x = d[(d.content == "err") & (d.model != "Argon")].assign(k=1, lk=0)
r = fit(x, f"y ~ C(model) * {T}"); wald(r, x, r"C\(model.*:C\(type", "err: Flash-Lite vs Qwen, model x type")

# ---------- 2.4 do the answers to one fact agree across types? ----------
print("\n=== 2.4 agreement across types: mean phi of the correct/incorrect indicators over type pairs, same fact ===")


def phi(a, b):
    return np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan


def mean_phi(x, ts):
    w = x.pivot_table(index=["rec", "fact"], columns="type", values="y")[ts].dropna()
    return np.nanmean([phi(w[a].values, w[b].values) for a, b in itertools.combinations(ts, 2)]), w


for m in LOAD:
    for c, ts in TYPES.items():
        x = d[(d.model == m) & (d.content == c)]
        if not len(x) or not set(ts) <= set(x.type):
            continue
        all_, w = mean_phi(x, ts)
        txt = f"  {m:10s} {c}: all facts {all_:+.3f}"
        if c == "err":
            for flag, lab in ((True, "mistake"), (False, "no mistake")):
                v, ww = mean_phi(x[x.mistake == flag], ts)
                txt += f" | {lab} only {v:+.3f}"
        print(txt)
