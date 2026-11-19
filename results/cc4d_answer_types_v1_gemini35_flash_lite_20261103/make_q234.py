"""Q2 / Q3 / Q4 figures and tables with chance / majority correction. Offline; run with python3.11 (matplotlib)."""

import collections
import json
import re

import matplotlib
import numpy as np

import merged

matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = merged.here
rng = np.random.default_rng(0)
B = 2000
N_OPTS = {"err": 6, "dur": 4, "loc": 5}  # single_choice options per row (letters seen in the data)
TYPES = ["bool", "single_choice", "multi_choice", "int", "open_text", "seconds"]
ROWS = ["err", "dur", "loc"]
SHORT = {"single_choice": "single", "multi_choice": "multi"}
ok = lambda r: r["correct"] is True  # noqa: E731


def rec(report_id):
    return re.match(r"cc4d-(\d+_\d+)__", report_id).group(1)


def cell(r):
    return r["field_id"][:3], r["extra"]["answer_type"]


def as_set(t):
    return set(t) if isinstance(t, list) else {t}


# ---- baselines, from the E1 facts (the same facts are asked in every condition) ----
e1 = [r for r in merged.load("fl_base_k1") if not r["extra"].get("not_done_item")]
tol = {}
for run in merged.runs():  # a field's seconds tolerance, from any run where it was answered
    for r in merged.load(run):
        if "tolerance" in r["extra"]:
            tol[r["report_id"].split("_half")[0], r["field_id"]] = r["extra"]["tolerance"]


def baseline(c):
    rr = [r for r in e1 if cell(r) == c]
    row, t = c
    n = len(rr)
    if t == "bool":
        maj = max(np.mean([r["target"] is True for r in rr]), np.mean([r["target"] is False for r in rr]))
        return {"chance": 0.5, "majority": maj}
    if t == "single_choice":
        letters = "ABCDEF"[: N_OPTS[row]]
        maj = max(np.mean([L in as_set(r["target"]) for r in rr]) for L in letters)
        return {"chance": float(np.mean([len(as_set(r["target"])) / N_OPTS[row] for r in rr])), "majority": maj}
    if t == "multi_choice":
        cnt = collections.Counter(json.dumps(sorted(r["target"])) for r in rr)
        return {"chance": None, "majority": cnt.most_common(1)[0][1] / n}
    if t == "int":
        cnt = collections.Counter(r["target"] for r in rr)
        return {"chance": None, "majority": cnt.most_common(1)[0][1] / n}
    if t == "open_text":  # always answering "none"
        return {"chance": None, "majority": float(np.mean([not r["target"] for r in rr]))}
    # seconds: best single constant answer (an oracle that never sees the video)
    med = float(np.median([v for (_, f), v in tol.items() if f.startswith(row)]))

    def tl(r):  # 13 fields were never answered validly: duration rule, or the row's median tolerance
        return tol.get((r["report_id"], r["field_id"])) or (max(2, 0.25 * r["target"]) if row == "dur" else med)

    pts = [(r["target"], tl(r)) for r in rr]
    grid = np.linspace(0, max(p for p, _ in pts) + 60, 600)
    best = max(np.mean([abs(g - p) <= tl for p, tl in pts]) for g in grid)
    return {"chance": None, "majority": float(best)}


CELLS = sorted({cell(r) for r in e1}, key=lambda c: (ROWS.index(c[0]), TYPES.index(c[1])))
BASE = {c: baseline(c) for c in CELLS}
BASE_USED = {c: max(v for v in BASE[c].values() if v is not None) for c in CELLS}
norm = lambda a, c: (a - BASE_USED[c]) / (1 - BASE_USED[c])  # noqa: E731


# ---- bootstrap by recording ----
def per_rec(rows, key):
    d = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = key(r)
        d[rec(r["report_id"]), k][0] += ok(r)
        d[rec(r["report_id"]), k][1] += 1
    return d


recs = sorted({rec(r["report_id"]) for r in e1})
RI = {r: i for i, r in enumerate(recs)}
draws = rng.integers(0, len(recs), size=(B, len(recs)))


def boot_acc(rows, c):
    """accuracy and its bootstrap samples for a list of rows"""
    num, den = np.zeros(len(recs)), np.zeros(len(recs))
    for r in rows:
        num[RI[rec(r["report_id"])]] += ok(r)
        den[RI[rec(r["report_id"])]] += 1
    acc = num.sum() / den.sum()
    samp = num[draws].sum(1) / np.maximum(den[draws].sum(1), 1)
    return acc, samp


def ci(x):
    return np.percentile(x, [2.5, 97.5])


# ================= Q2 / Q3 : E1 =================
E = {}
for c in CELLS:
    acc, samp = boot_acc([r for r in e1 if cell(r) == c], c)
    E[c] = (acc, samp, norm(acc, c), norm(samp, c))

colors = {"bool": "C0", "single_choice": "C1", "multi_choice": "C2", "int": "C3", "open_text": "C4", "seconds": "C5"}

fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for a, row in zip(ax, ROWS):
    cs = [c for c in CELLS if c[0] == row]
    for i, c in enumerate(cs):
        n, lo_hi = E[c][2], ci(E[c][3])
        a.errorbar(i, n, yerr=[[n - lo_hi[0]], [lo_hi[1] - n]], fmt="o", color=colors[c[1]], capsize=3)
    a.axhline(0, color="k", ls="--", lw=1)
    a.set_xticks(range(len(cs))); a.set_xticklabels([SHORT.get(c[1], c[1]) for c in cs], fontsize=8)
    a.set_title(row)
ax[0].set_ylabel("chance-normalized accuracy\n(0 = majority / chance)")
plt.tight_layout(); plt.savefig(f"{here}/figs/q2_type.png", dpi=130); plt.close()

q2 = ["| 內容 | 型別 | 準確率 | 基準（chance / majority） | 校正後 [95% CI] |", "| --- | --- | --- | --- | --- |"]
for c in CELLS:
    b = BASE[c]
    base_s = " / ".join(f"{v:.2f}" if v is not None else "–" for v in (b["chance"], b["majority"]))
    lo, hi = ci(E[c][3])
    q2.append(f"| {c[0]} | {SHORT.get(c[1], c[1])} | {E[c][0]:.3f} | {base_s} | {E[c][2]:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
print("\n".join(q2))

# paired difference of normalized accuracy vs bool, within a row
print("\nQ2 差異（校正後，型別 − bool，同一事實配對、依錄影 bootstrap）")
for row in ROWS:
    for c in [c for c in CELLS if c[0] == row and c[1] != "bool"]:
        d = E[c][3] - E[(row, "bool")][3]
        lo, hi = ci(d)
        print(f"{row} {SHORT.get(c[1], c[1])} − bool: {E[c][2] - E[(row, 'bool')][2]:+.2f} [{lo:+.2f}, {hi:+.2f}]")

# error-row bool / single split by mistake / no mistake
print("\nerr 拆分（準確率）")
for t in ("bool", "single_choice", "multi_choice", "int", "open_text"):
    rr = [r for r in e1 if cell(r) == ("err", t)]
    def has_mistake(r):
        tg = r["target"]
        return tg is True if t == "bool" else (tg is False if False else None)
    if t == "bool":  # target true = no mistake (the step was done right)
        a = [ok(r) for r in rr if r["target"] is False]
        b = [ok(r) for r in rr if r["target"] is True]
        print("bool: 有錯步驟", round(np.mean(a), 3), len(a), "| 無錯步驟", round(np.mean(b), 3), len(b))
    else:
        none = [r for r in rr if (r["target"] == ["F"] if t == "multi_choice" else (r["target"] == [] if t == "open_text" else (r["target"] == 0 if t == "int" else as_set(r["target"]) == {"F"})))]
        rest = [r for r in rr if r not in none]
        print(t, "有錯步驟", round(np.mean([ok(r) for r in rest]), 3), len(rest), "| 無錯步驟", round(np.mean([ok(r) for r in none]), 3), len(none))

# ================= Q3 =================
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [2, 1]})
w = 0.38
for j, t in enumerate(["bool", "single_choice"]):
    for i, row in enumerate(ROWS):
        c = (row, t); n = E[c][2]; lo, hi = ci(E[c][3])
        ax[0].bar(i + (j - 0.5) * w, n, w, color=colors[t], yerr=[[n - lo], [hi - n]], capsize=3, label=SHORT.get(t, t) if i == 0 else None)
ax[0].axhline(0, color="k", ls="--", lw=1); ax[0].set_xticks(range(3)); ax[0].set_xticklabels(ROWS)
ax[0].set_ylabel("chance-normalized accuracy"); ax[0].legend(fontsize=8); ax[0].set_title("bool and single (in every row)")
for i, row in enumerate(["dur", "loc"]):
    c = (row, "seconds"); n = E[c][2]; lo, hi = ci(E[c][3])
    ax[1].bar(i, n, 0.5, color=colors["seconds"], yerr=[[n - lo], [hi - n]], capsize=3)
ax[1].axhline(0, color="k", ls="--", lw=1); ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["dur", "loc"]); ax[1].set_title("seconds (descriptive)")
plt.tight_layout(); plt.savefig(f"{here}/figs/q3_content.png", dpi=130); plt.close()

print("\nQ3 內容間差異（校正後，bool / single，行 − err）")
for t in ("bool", "single_choice"):
    for row in ("dur", "loc"):
        d = E[(row, t)][3] - E[("err", t)][3]
        lo, hi = ci(d)
        print(f"{t} {row} − err: {E[(row, t)][2] - E[('err', t)][2]:+.2f} [{lo:+.2f}, {hi:+.2f}]")
d = E[("loc", "seconds")][3] - E[("dur", "seconds")][3]
print(f"seconds loc − dur: {E[('loc', 'seconds')][2] - E[('dur', 'seconds')][2]:+.2f} [{ci(d)[0]:+.2f}, {ci(d)[1]:+.2f}]")

# ================= Q4 : E4 vs E3, same order =================
def index(run):
    return {(r["call_id"].split("::")[0], r["field_id"]): r for r in merged.load(run)}


def e4_pairs(e4_run, e3_run):
    e3 = index(e3_run)
    out = collections.defaultdict(list)
    for r in merged.load(e4_run):
        rep = r["call_id"].split("::")[0]
        base_rep = re.sub(r"_half\d$", "", rep)
        grp = base_rep.split("__")[1]  # e.g. err_bool
        if not grp.endswith(r["field_id"].split("_", 1)[1]):
            continue  # a filler question, not scored here
        c = cell(r)
        out[c].append((e3[base_rep, r["field_id"]], r))
    return out


Q4 = {}
for tag, e4, e3 in [("fwd", "fl_e4_k8", "fl_base_k8"), ("rev", "fl_e4_rev_k8", "fl_base_rev_k8")]:
    for c, pairs in e4_pairs(e4, e3).items():
        num3, num4, den = (np.zeros(len(recs)) for _ in range(3))
        rw = wr = 0
        for a, b in pairs:
            i = RI[rec(a["report_id"])]
            num3[i] += ok(a); num4[i] += ok(b); den[i] += 1
            rw += ok(a) and not ok(b); wr += ok(b) and not ok(a)
        a3, a4 = num3.sum() / den.sum(), num4.sum() / den.sum()
        d = (num4[draws].sum(1) - num3[draws].sum(1)) / np.maximum(den[draws].sum(1), 1)
        Q4[tag, c] = {"e3": a3, "e4": a4, "delta": a4 - a3, "ci": ci(d), "rw": int(rw), "wr": int(wr), "n": int(den.sum()),
                      "n3": norm(a3, c), "n4": norm(a4, c), "dn": (a4 - a3) / (1 - BASE_USED[c])}

fig, ax = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
for a, t in zip(ax.flat, TYPES):
    rows_t = [row for row in ROWS if (row, t) in BASE]
    for i, row in enumerate(rows_t):
        c = (row, t)
        for k, tag in enumerate(("fwd", "rev")):
            q = Q4[tag, c]; x = i + (k - 0.5) * 0.3
            a.plot([x, x], [q["n3"], q["n4"]], color="gray", lw=1)
            a.plot(x, q["n3"], "o", color="k", ms=5, label="single type (E3)" if i == 0 and k == 0 else None)
            a.plot(x, q["n4"], "s", color=colors[t], ms=5, label="mixed (E4)" if i == 0 and k == 0 else None)
    a.set_xticks(range(len(rows_t))); a.set_xticklabels(rows_t); a.set_title(SHORT.get(t, t)); a.axhline(0, color="k", ls="--", lw=.8)
    a.set_xlim(-.6, len(rows_t) - .4)
    if t == "bool":
        a.legend(fontsize=7)
ax[0, 0].set_ylabel("chance-normalized accuracy"); ax[1, 0].set_ylabel("chance-normalized accuracy")
fig.suptitle("each pair of points: left = forward order, right = reversed order", fontsize=9)
plt.tight_layout(); plt.savefig(f"{here}/figs/q4_mixed.png", dpi=130); plt.close()

q4 = ["| 順序 | 內容 | 型別 | n | 單一型別 | 混合 | 差（pt）[95% CI] | r→w | w→r |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for tag in ("fwd", "rev"):
    for c in CELLS:
        q = Q4[tag, c]
        q4.append(f"| {tag} | {c[0]} | {SHORT.get(c[1], c[1])} | {q['n']} | {q['e3']:.3f} | {q['e4']:.3f} | {100*q['delta']:+.1f} [{100*q['ci'][0]:+.1f}, {100*q['ci'][1]:+.1f}] | {q['rw']} | {q['wr']} |")
print("\n" + "\n".join(q4))
for tag in ("fwd", "rev"):
    rw = sum(Q4[tag, c]["rw"] for c in CELLS); wr = sum(Q4[tag, c]["wr"] for c in CELLS)
    print(tag, "pooled r→w", rw, "w→r", wr, "z", round((wr - rw) / np.sqrt(rw + wr), 2))

# pooled over the 11 cells, bootstrap by recording (items from one video are not independent)
for tag, e4, e3 in [("fwd", "fl_e4_k8", "fl_base_k8"), ("rev", "fl_e4_rev_k8", "fl_base_rev_k8")]:
    n3, n4, dn = (np.zeros(len(recs)) for _ in range(3))
    for c, pairs in e4_pairs(e4, e3).items():
        for a, b in pairs:
            i = RI[rec(a["report_id"])]
            n3[i] += ok(a); n4[i] += ok(b); dn[i] += 1
    d = (n4[draws].sum(1) - n3[draws].sum(1)) / dn[draws].sum(1)
    print(tag, "pooled delta", f"{100*(n4.sum()-n3.sum())/dn.sum():+.2f}pt [{100*ci(d)[0]:+.2f}, {100*ci(d)[1]:+.2f}]")
