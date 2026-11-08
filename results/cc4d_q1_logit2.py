"""Q1 follow-up: (A) model x content x k and model x type x k three-way interactions, (B) position,
(C) answer bias on bool, (D) video length. Reuses the loaders of cc4d_q1_logit.py. Same conventions: base group,
filmed items, SE clustered by recording, lk = log2(question count)."""
import json, re
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from cc4d_q1_logit import LOAD, SHORT, fit, show, ame, here

DATA = f"{here}/../data/cc4d_answer_types_v1"
_pos, _len = {}, {}


def meta(data_group):
    """(report, field) -> index in report; report -> video minutes"""
    if data_group not in _pos:
        for l in open(f"{DATA}/{data_group}/reports.jsonl"):
            r = json.loads(l)
            _len[r["id"]] = sum(v["time_range"]["end"] - v["time_range"]["start"] for v in r["input"]["videos"]) / 60
            for i, f in enumerate(r["fields"]):
                _pos[data_group, r["id"], f["id"]] = i
    return data_group


def table(groups):
    out = []
    for g in groups:
        k = int(re.search(r"k(\d+)$", g).group(1)); dg = meta(re.sub(r"_k\d+$", "", g)); rev = "rev" in g
        for m, f in LOAD.items():
            try:
                rows = f(g)
            except Exception:
                continue
            for r in rows:
                if r["extra"].get("not_done_item") or r["correct"] is None:
                    continue
                i = _pos[dg, r["report_id"], r["field_id"]] % k  # index in call; reversed groups are stored already reversed
                out.append(dict(model=m, k=k, rev=int(rev), rec=re.match(r"cc4d-(\d+_\d+)__", r["report_id"]).group(1),
                                fact=(r["report_id"], r["field_id"]), content=r["field_id"][:3],
                                type=SHORT.get(r["extra"]["answer_type"], r["extra"]["answer_type"]),
                                y=int(r["correct"] is True), pred=r["predicted"], target=r["target"],
                                posfrac=0.0 if k == 1 else i / (k - 1), vmin=_len[r["report_id"]]))
    d = pd.DataFrame(out); d["cell"] = d.content + " " + d.type
    return d


def wald(res, df, pattern, label):
    """joint Wald test of all coefficients whose name matches pattern"""
    idx = [i for i, n in enumerate(res.params.index) if re.search(pattern, n)]
    w = res.wald_test(np.eye(len(res.params))[idx], scalar=True)
    print(f"  Wald {label}: chi2={float(w.statistic):.2f} df={len(idx)} p={float(w.pvalue):.4f}")


if __name__ == "__main__":
    d = table(["base_k1", "base_k4", "base_k8", "base_rev_k4", "base_rev_k8"])
    fw = d[d.rev == 0]
    fw = fw[fw.groupby(["model", "fact"]).k.transform("nunique") == 3]
    common = ["dur bool", "dur single", "loc bool", "loc single", "loc seconds"]
    M = "C(model, Treatment('Argon'))"

    print("=== A1. model x content x k (bool+single; dur vs loc, all three models) ===")
    a = fw[fw.type.isin(["bool", "single"]) & (fw.content != "err")]
    r = fit(a, f"y ~ lk * {M} * C(content, Treatment('dur')) + C(type)")
    wald(r, a, r"lk:.*:.*", "3-way (model x content x k)"); wald(r, a, r"^lk:C\(model", "model x k")
    print(ame(r, a, ["model", "content"]).to_string())
    print("--- adding err, Flash-Lite vs Qwen only ---")
    a = fw[fw.type.isin(["bool", "single"]) & (fw.model != "Argon")]
    r = fit(a, "y ~ lk * C(model, Treatment('Qwen')) * C(content, Treatment('err')) + C(type)")
    wald(r, a, r"lk:.*:.*", "3-way (model x content x k)")
    print(ame(r, a, ["model", "content"]).to_string())

    print("\n=== A2. model x type x k (dur+loc; bool, single, seconds; all three models) ===")
    a = fw[fw.type.isin(["bool", "single", "seconds"]) & (fw.content != "err")]
    r = fit(a, f"y ~ lk * {M} * C(type, Treatment('bool')) + C(content)")
    wald(r, a, r"lk:.*:.*", "3-way (model x type x k)")
    print(ame(r, a, ["model", "type"]).to_string())
    print("--- error row (bool, single, multi, int), Flash-Lite vs Qwen ---")
    a = fw[(fw.content == "err") & fw.type.isin(["bool", "single", "multi", "int", "open_text"])]
    r = fit(a, "y ~ lk * C(model, Treatment('Qwen')) * C(type, Treatment('bool'))")
    wald(r, a, r"lk:.*:.*", "3-way (model x type x k)")
    print(ame(r, a, ["model", "type"]).to_string())

    print("\n=== B. position in the call (posfrac 0 = first, 1 = last; forward + reversed, k = 4, 8) ===")
    p = d[(d.k > 1) & d.cell.isin(common)]
    for m in LOAD:
        x = p[p.model == m]
        show(f"{m}: position effect", fit(x, "y ~ lk + posfrac + cell"), [r"posfrac"])
    r = fit(p, f"y ~ lk + posfrac * {M} + cell"); wald(r, p, r"posfrac:", "position slope differs by model")
    print("--- does adjusting for position change the k effect? forward only, k = 1/4/8 (k=1 coded posfrac 0) ---")
    f5 = fw[fw.cell.isin(common)]
    for m in LOAD:
        x = f5[f5.model == m]
        print(f"  {m}: lk {fit(x, 'y ~ lk + cell').params['lk']:+.3f} -> {fit(x, 'y ~ lk + posfrac + cell').params['lk']:+.3f}"
              f"  (posfrac {fit(x, 'y ~ lk + posfrac + cell').params['posfrac']:+.3f})")

    print("\n=== C. answer bias on bool: P(answer = yes) and accuracy split by true answer ===")
    b = fw[fw.type == "bool"].copy()
    b = b[b.pred.isin([True, False])]; b["yes"] = b.pred.astype(int); b["tgt"] = b.target.astype(int)
    for m in LOAD:
        x = b[b.model == m]
        print(f"  {m}: yes-rate by k", {k: round(v, 3) for k, v in x.groupby('k').yes.mean().items()},
              "| yes-rate when truth=yes", {k: round(v, 3) for k, v in x[x.tgt == 1].groupby('k').yes.mean().items()},
              "| truth=no", {k: round(v, 3) for k, v in x[x.tgt == 0].groupby('k').yes.mean().items()})
        res = fit(x.assign(y=x.yes), "y ~ lk * tgt + cell")
        show(f"{m}: P(yes) ~ k x truth", res, [r"^lk"])

    print("\n=== D. video length (minutes) ===")
    L = fw[fw.cell.isin(common)].copy(); L["v10"] = (L.vmin - L.vmin.mean()) / 10
    L["cls"] = pd.cut(L.vmin, [0, 11, 17, 99], labels=["A", "B", "C"])
    for m in LOAD:
        x = L[L.model == m]
        show(f"{m}: k x length(10 min)", fit(x, "y ~ lk * v10 + cell"), [r"v10"])
        print("  AME per doubling (pt) by class:\n", ame(fit(x, "y ~ lk * C(cls) + cell"), x, ["cls"]).to_string())
