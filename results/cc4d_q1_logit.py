"""Q1 (question count) across Flash-Lite, Qwen3-VL-32B and Argon: long table + logistic regressions.
Run: python cc4d_q1_logit.py   (needs pandas, statsmodels). Base group, forward order, filmed items,.
Standard errors are clustered by recording (same unit as the bootstrap in the other analyses)."""
import glob, importlib.util, json, os, re
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

here = os.path.dirname(os.path.abspath(__file__))
SHORT = {"single_choice": "single", "multi_choice": "multi"}


def module(rel, name):
    s = importlib.util.spec_from_file_location(name, f"{here}/{rel}")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


qw = module("cc4d_answer_types_v1_qwen3vl_32b_20261107/merged.py", "qw")
gm = module("cc4d_answer_types_v1_gemini35_flash_lite_20261103/merged.py", "gm")
ARG = f"{here}/cc4d_answer_types_v1_gemini4_argon_planA_20261108/runs"


def argon(run):
    return [json.loads(x) for x in open(sorted(glob.glob(f"{ARG}/ar_{run}/evaluations/2*/field_scores.jsonl"))[-1])]


LOAD = {"Flash-Lite": lambda g: gm.load("fl_" + g), "Qwen": lambda g: qw.load_all("fl_" + g), "Argon": argon}


def table(groups, forward_k):
    out = []
    for m, f in LOAD.items():
        for g, k in zip(groups, forward_k):
            try:
                rows = f(g)
            except Exception:
                continue
            for r in rows:
                if r["extra"].get("not_done_item") or r["correct"] is None:
                    continue
                out.append(dict(model=m, k=k, rec=re.match(r"cc4d-(\d+_\d+)__", r["report_id"]).group(1),
                                fact=(r["report_id"], r["field_id"]), content=r["field_id"][:3],
                                type=SHORT.get(r["extra"]["answer_type"], r["extra"]["answer_type"]),
                                y=int(r["correct"] is True)))
    return pd.DataFrame(out)


def fit(df, formula):
    d = df.copy(); d["lk"] = np.log2(d.k)  # one unit = doubling the question count
    d["rid"] = pd.factorize(d.rec)[0]
    return smf.logit(formula, d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.rid})


def show(title, res, keep):
    print(f"\n## {title}  (n={int(res.nobs)})")
    for n in res.params.index:
        if any(re.search(p, n) for p in keep):
            lo, hi = res.conf_int().loc[n]
            print(f"  {n:38s} coef {res.params[n]:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={res.pvalues[n]:.3f}  OR={np.exp(res.params[n]):.2f}")


def ame(res, df, by):
    """Average change in accuracy (pt) per doubling of k, by group, from the fitted model."""
    d = df.copy(); d["lk"] = np.log2(d.k); d["rid"] = 0
    p0 = res.predict(d.assign(lk=0)); p1 = res.predict(d.assign(lk=1))
    return (100 * (p1 - p0)).groupby([d[b] for b in by]).mean().round(2)


if __name__ == "__main__":
    main = table(["base_k1", "base_k4", "base_k8"], [1, 4, 8])
    e5 = pd.concat([table([f"e5_{c}_base_k{k}" for k in (1, 4, 8, 16)], [1, 4, 8, 16]).assign(e5=c) for c in ("dur", "loc")])
    common = ["dur bool", "dur single", "loc bool", "loc single", "loc seconds"]
    main["cell"] = main.content + " " + main.type
    c5 = main[main.cell.isin(common)]
    # keep only facts present at all three k for the same model (paired design)
    c5 = c5[c5.groupby(["model", "fact"]).k.transform("nunique") == 3]
    main = main[main.groupby(["model", "fact"]).k.transform("nunique") == 3]
    print("facts per model/k:\n", c5.groupby(["model", "k"]).size().unstack())

    print("\n=== 1. accuracy ~ log2(k), each model, 5 common cells ===")
    for m in LOAD:
        show(f"{m}", fit(c5[c5.model == m], "y ~ lk"), ["lk"])
    print("\n--- 10 cells (Flash-Lite, Qwen) ---")
    for m in ("Flash-Lite", "Qwen"):
        show(f"{m}", fit(main[main.model == m], "y ~ lk"), ["lk"])
    print("\n--- model difference in slope (5 common cells, pooled) ---")
    r = fit(c5, "y ~ lk * C(model, Treatment('Argon')) + cell")
    show("lk x model", r, ["lk"])

    print("\n=== 2. content x k (bool + single only: the types present in all contents) ===")
    for m in ("Flash-Lite", "Qwen"):
        d = main[(main.model == m) & main.type.isin(["bool", "single"])]
        r = fit(d, "y ~ lk * C(content, Treatment('err')) + C(type)")
        show(f"{m}: k effect depends on content?", r, ["lk"])
        print("  AME per doubling (pt) by content:\n", ame(r, d, ["content"]).to_string())
    d = c5[c5.model == "Argon"]
    r = fit(d, "y ~ lk * C(content, Treatment('dur')) + C(type)")
    show("Argon (dur vs loc only)", r, ["lk"]); print("  AME by content:\n", ame(r, d, ["content"]).to_string())

    print("\n=== 3. type x k (dur + loc rows: bool, single, seconds) ===")
    for m in LOAD:
        d = main[(main.model == m) & main.type.isin(["bool", "single", "seconds"]) & (main.content != "err")] if m != "Argon" else c5[c5.model == m]
        r = fit(d, "y ~ lk * C(type, Treatment('bool')) + C(content)")
        show(f"{m}: k effect depends on type?", r, ["lk"]); print("  AME by type:\n", ame(r, d, ["type"]).to_string())
    print("\n--- error row, Qwen / Flash-Lite: bool, single, multi, int ---")
    for m in ("Flash-Lite", "Qwen"):
        d = main[(main.model == m) & (main.content == "err")]
        r = fit(d, "y ~ lk * C(type, Treatment('bool'))")
        show(m, r, ["lk"]); print("  AME by type:\n", ame(r, d, ["type"]).to_string())

    print("\n=== 4a. E5: k up to 16, per model, by content (bool/single/seconds) ===")
    e5 = e5[e5.groupby(["model", "fact"]).k.transform("nunique") == 4]
    for m in ("Flash-Lite", "Qwen"):
        for c in ("dur", "loc"):
            d = e5[(e5.model == m) & (e5.e5 == c)]
            if len(d):
                r = fit(d, "y ~ lk * C(type, Treatment('bool'))")
                show(f"{m} {c}", r, ["lk"]); print("  AME by type:\n", ame(r, d, ["type"]).to_string())
