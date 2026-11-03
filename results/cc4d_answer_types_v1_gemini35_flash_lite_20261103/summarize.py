"""Rebuild summary.json and print a table from runs/*/evaluations/*/metrics.json (offline)."""

import glob
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
rows = []
for d in sorted(glob.glob(f"{here}/runs/*/")):
    name = os.path.basename(d.rstrip("/"))
    m = json.load(open(sorted(glob.glob(d + "evaluations/*/metrics.json"))[-1]))
    f, u = m["fields"], m["usage"]
    rows.append(
        {
            "run": name,
            "calls": m["execution"]["execution_completion"]["denominator"],
            "fields": f["field_accuracy"]["denominator"],
            "accuracy": f["field_accuracy"]["value"],
            "valid_answer_rate": f["valid_answer_rate"]["value"],
            "by_answer_type": {k: v["value"] for k, v in f["by_answer_type"].items()},
            "hallucination": f["not_done"]["hallucination"]["value"],
            "false_not_done": f["not_done"]["false_not_done"]["value"],
            "cost_usd": u["cost"],
            "cost_status": u["cost_status"],
        }
    )
json.dump(rows, open(f"{here}/summary.json", "w"), indent=1)
print(f"{'run':28}{'calls':>7}{'fields':>8}{'acc':>7}{'valid':>7}{'cost':>8}")
for r in rows:
    print(
        f"{r['run']:28}{r['calls']:>7}{r['fields']:>8}{r['accuracy']:>7.2f}"
        f"{r['valid_answer_rate']:>7.3f}{r['cost_usd']:>8.2f}"
    )
print("total cost", round(sum(r["cost_usd"] for r in rows), 2), "calls", sum(r["calls"] for r in rows))
