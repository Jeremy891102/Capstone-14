"""Score the attention-intervention outputs (scripts/qwen_intervene.py) against CC4D ground truth.

python results/cc4d_intervene_score.py RUN_DIR GT.jsonl ROWS.json.gz OUT.json INTERVENE.jsonl [...]
RUN_DIR: the finished vLLM run that was regenerated (k32_fwd_kall); GT: data/cc4d_k32_v1/fwd/ground_truth.jsonl;
ROWS: rows_qwen32.json.gz (vLLM k = 8 / k = all correctness, same scorer); OUT: per-question scores.

Every generated text goes through the same parser and scorer as `video_report evaluate`. Accuracy is on
ordinary questions (abstention questions excluded), pooled over questions, with 95% bootstrap intervals over
reports. Differences are paired by question: each condition minus `base` (the manual-path alpha = 1 run),
plus `base` minus vLLM k = 8 for scale. Also printed: the mean late-layer video share of attention,
the share of calls whose answer JSON parsed, and the output length.
"""

import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from video_report.datasets.base import load_ground_truth
from video_report.evaluation.parser import parse_response
from video_report.evaluation.scorers import ScoringConfig, score_field
from video_report.run_store import RunDir

run_dir, gt_path, rows_path, out_path, *files = sys.argv[1:]
rng = np.random.default_rng(0)
run = RunDir(Path(run_dir))
reports = {r.id: r for r in run.load_reports()}
selection = {r.id: [f.id for f in r.fields] for r in reports.values()}
targets = {(t.report_id, t.field_id): t for t in load_ground_truth(Path(gt_path))}
cfg = ScoringConfig()

recs = [json.loads(line) for f in files for line in open(f)]
recs = [r for r in recs if "condition" in r]
conds = list(dict.fromkeys(r["condition"] for r in recs))
print(f"{len(recs)} records, {len({r['call_id'] for r in recs})} calls, conditions {conds}")

score, parsed_ok, late, outlen = {}, collections.defaultdict(list), collections.defaultdict(list), {}
for r in recs:
    c = r["condition"]
    parsed = parse_response(r["text"], r["field_ids"], allow_markdown_fence=False)
    parsed_ok[c].append(parsed.status == "ok")
    if r["late_video_share"] is not None:
        late[c].append(r["late_video_share"] * 100)
    outlen.setdefault(c, []).append(r["out_tokens"])
    for fid in r["field_ids"]:
        s = score_field(
            reports[r["report_id"]].field(fid),
            targets[r["report_id"], fid],
            r["call_id"],
            parsed,
            None,
            config=cfg,
            judge=None,
        ).to_json()
        score[c, r["report_id"], fid] = (int(s["correct"]), s["extra"]["not_done_item"], s["extra"]["answer_type"])

# vLLM rows (k = 8 / k = all), same questions
for r in json.load(gzip.open(rows_path, "rt")):
    if r[0] == "fwd" and r[1] in ("8", "all"):
        score[f"vllm_k{r[1]}", r[2], r[3]] = (r[5], r[6], None)
names = [*conds, "vllm_kall", "vllm_k8"]
qs = sorted({(k[1], k[2]) for k in score if k[0] == "base"})
qs = [q for q in qs if all((n, *q) in score for n in names) and score["base", *q][1] == 0]
by_rep = collections.defaultdict(list)
for q in qs:
    by_rep[q[0]].append(q)
reps = sorted(by_rep)
print(f"{len(qs)} ordinary questions with every condition scored, {len(reps)} reports\n")


def ok(n, sub):
    return np.array([score[n, *q][0] for q in sub], float) * 100


def boot(fn, B=1000):
    est = fn([q for r in reps for q in by_rep[r]])
    bs = [fn([q for r in rng.choice(reps, len(reps)) for q in by_rep[r]]) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def fmt(x, signed=False):
    return f"{x[0]:+6.1f} [{x[1]:+6.1f}, {x[2]:+6.1f}]" if signed else f"{x[0]:5.1f} [{x[1]:5.1f}, {x[2]:5.1f}]"


print(f"{'condition':11}{'accuracy %':>22}{'minus base (pp)':>26}{'late video %':>14}{'parsed':>8}{'tokens':>8}")
summary = {}
for n in names:
    acc = boot(lambda sub, n=n: ok(n, sub).mean())
    dif = boot(lambda sub, n=n: (ok(n, sub) - ok("base", sub)).mean()) if n != "base" else None
    summary[n] = dict(acc=acc, diff=dif)
    lv = f"{np.mean(late[n]):.1f}" if late.get(n) else "-"
    pr = f"{100 * np.mean(parsed_ok[n]):.0f}%" if n in parsed_ok else "-"
    ol = f"{np.mean(outlen[n]):.0f}" if n in outlen else "-"
    print(f"{n:11}{fmt(acc):>22}{fmt(dif, True) if dif else '':>26}{lv:>14}{pr:>8}{ol:>8}")

print("\nby answer type (accuracy %, minus base)")
types = sorted({score["base", *q][2] for q in qs})
for t in types:
    sub = [q for q in qs if score["base", *q][2] == t]
    print(f"  {t:13} n={len(sub):4d}  " + "  ".join(
        f"{n}:{ok(n, sub).mean():.1f}" + (f"({(ok(n, sub) - ok('base', sub)).mean():+.1f})" if n != "base" else "")
        for n in conds
    ))

Path(out_path).write_text(json.dumps({"summary": summary, "n": len(qs), "reports": len(reps)}, default=float))
