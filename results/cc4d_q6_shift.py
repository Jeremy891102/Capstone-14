"""Q6 diagnostic: for single-choice questions of Qwen (E5 localization), how often does the predicted letter of question i match the
target of question i + delta in the same call? Elevated matches at delta = +-1 mean answers shifted by one position (misalignment)."""
import collections, json, os
import numpy as np
from cc4d_q1_logit import qw
import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/cc4d_answer_types_v1")
def order(g):
    o={}
    for l in open(f"{D}/{g}/reports.jsonl"):
        r=json.loads(l); o[r["id"]]=[f["id"] for f in r["fields"]]
    return o
asset=lambda t:set(t) if isinstance(t,list) else {t}
for grp,dg in (("e5_loc_base_rev_k16","e5_loc_base_rev"),("e5_loc_base_k16","e5_loc_base"),("e5_loc_base_rev_k8","e5_loc_base_rev"),("e5_loc_base_k8","e5_loc_base")):
    o=order(dg); rows=qw.load_all("fl_"+grp)
    by=collections.defaultdict(dict)
    for r in rows:
        if r["extra"]["answer_type"]!="single_choice" or r["extra"].get("not_done_item"): continue
        by[r["report_id"]][r["field_id"]]=r
    res=collections.defaultdict(list)
    for rep,d in by.items():
        fs=[f for f in o[rep] if f in d]
        for i,f in enumerate(fs):
            for delta in (-2,-1,0,1,2):
                j=i+delta
                if 0<=j<len(fs):
                    res[delta].append(str(d[f]["predicted"]) in {str(x) for x in asset(d[fs[j]]["target"])})
    print(grp, {k: round(float(np.mean(v)),3) for k,v in sorted(res.items())}, "n", len(res[0]))
