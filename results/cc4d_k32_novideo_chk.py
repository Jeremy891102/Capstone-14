"""usage: python results/cc4d_k32_novideo_chk.py RUNS_DIR PREFIX  -> JSON summary to stdout (run from repo root)"""
import glob, json, sys, collections
runs, pre = sys.argv[1:3]
out = {}
for o in ("fwd", "rev"):
    for k in ("8", "16", "32", "all"):
        r = f"{runs}/{pre}_{o}_k{k}"
        ev = sorted(glob.glob(f"{r}/evaluations/2*"))[-1]
        cp = [json.loads(x) for x in open(ev + "/call_parses.jsonl")]
        fs = [json.loads(x) for x in open(ev + "/field_scores.jsonl")]
        d = out.setdefault(k, collections.Counter())
        d["calls"] += len(cp)
        for c in cp:
            d["parse_" + c["parse_status"]] += 1
            d["missing_fields"] += len(c["missing"]); d["extra_keys"] += len(c["extra_keys"])
        fin = collections.Counter(); L = []; 
        for f in glob.glob(f"{r}/responses/*/attempt-*.json"):
            j = json.load(open(f)); fin[j["response"]["finish_reason"]] += 1; L.append(len(j["response"]["text"] or ""))
        for a, b in fin.items(): d["finish_" + a] += b
        d["text_chars_sum"] += sum(L); d["responses"] += len(L)
        for s in fs:
            x = s["extra"]; f = s["field_id"]; p = s["predicted"]
            d["outcome_" + s["outcome"]] += 1
            ab = "abs_" if x["not_done_item"] else "norm_"
            if f.endswith("_correct"):
                d[ab + "correct_pred_" + str(p)] += 1; d[ab + "correct_tgt_" + str(s["target"])] += 1
            elif f.endswith("_error_types"):
                key = "none" if p == [] else ("null" if p is None else ("list" if isinstance(p, list) else str(p)))
                d[ab + "err_pred_" + key] += 1
                t = s["target"]; d[ab + "err_tgt_" + ("none" if t == [] else str(t) if t == "not_visible" else "some")] += 1
            elif f.endswith("_duration") and not x["not_done_item"]:
                if isinstance(p, (int, float)): d["dur_n"] += 1; d["dur_sum_log_ok"] += 1; d.setdefault("durs", []) if False else None
                d["dur_pred_" + ("10" if p == 10 else "other")] += 1
print(json.dumps({k: dict(v) for k, v in out.items()}, indent=0, sort_keys=True))
