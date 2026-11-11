"""Build rows.json.gz (format in cc4d_k32.py) from evaluated k32 runs.
python results/cc4d_k32_rows.py RUNS_DIR OUT.json.gz   (run from the repo root; needs data/cc4d_k32_v1)"""
import glob, gzip, json, re, sys

runs, out = sys.argv[1:3]
rows = []
for o in ("fwd", "rev"):
    gt = {(g["report_id"], g["field_id"]): g["target"] for g in map(json.loads, open(f"data/cc4d_k32_v1/{o}/ground_truth.jsonl"))}
    for k in ("8", "16", "32", "all"):
        r = f"{runs}/k32_{o}_k{k}"
        fs = [json.loads(x) for x in open(sorted(glob.glob(f"{r}/evaluations/2*/field_scores.jsonl"))[-1])]
        pos = {}  # (call_id, field) -> (index from end, call length)
        for c in glob.glob(f"{r}/calls/*.json"):
            c = json.load(open(c)); n = len(c["field_ids"])
            for i, f in enumerate(c["field_ids"]): pos[c["call_id"], f] = (n - 1 - i, n)
        for s in fs:
            f, x = s["field_id"], s["extra"]
            end, n = pos[s["call_id"], f]
            nv = s["predicted"] == "not_visible" or "'not_visible'" in (s["detail"] or "")
            t = gt[s["report_id"], f]
            nm = (1 if t == [] else 0) if f.endswith("_error_types") and not x["not_done_item"] else -1
            rows.append([o, k, s["report_id"], f, x["answer_type"], int(s["correct"]), int(x["not_done_item"]), int(nv), end, n, nm])
json.dump(rows, gzip.open(out, "wt"))
print(len(rows), "rows")
