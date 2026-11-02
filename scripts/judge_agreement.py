# ruff: noqa: E501
"""Check the open_text judge against human labels (docs/scoring_rules.md, "Validation").

  sample   pick open_text answers the judge graded from an evaluation, write a sheet to label
  compare  read the filled sheet, report agreement with the judge and list the disagreements

    python scripts/judge_agreement.py sample --run RUN --eval EVAL_DIR --out sheet.csv [-n 100]
    python scripts/judge_agreement.py compare --sheet sheet.csv --key sheet.key.json

The sheet never shows the judge's verdict. Fill the `human` column with `correct` or `incorrect`.
Answers that were just "none" never reach the judge and are left out.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from video_report.run_store import RunDir


def sample(args: argparse.Namespace) -> None:
    reports = {r.id: r for r in RunDir(Path(args.run)).load_reports()}
    rows = [
        json.loads(x) for x in (Path(args.eval) / "field_scores.jsonl").read_text().splitlines()
    ]
    judged = [r for r in rows if "extra_claims" in r["extra"]]  # only answers the judge saw
    if not judged:
        sys.exit("no judged open_text answers in this evaluation")
    gt = {}
    for line in Path(args.ground_truth).read_text().splitlines():
        g = json.loads(line)
        gt[g["report_id"], g["field_id"]] = g
    rng = random.Random(args.seed)
    # half with a "none" reference and half with mistakes, so neither kind is missed
    none = [r for r in judged if not r["target"]]
    mist = [r for r in judged if r["target"]]
    k = min(args.n // 2, len(none), len(mist))
    pick = rng.sample(none, min(args.n - k, len(none)) if len(mist) < k else k)
    pick += rng.sample(mist, min(args.n - len(pick), len(mist)))
    rng.shuffle(pick)
    sheet, key = [], []
    for i, r in enumerate(pick, 1):
        spec = reports[r["report_id"]].field(r["field_id"])
        ev = gt[r["report_id"], r["field_id"]]["evidence"]
        sheet.append(
            {
                "n": i,
                "question": spec.question.split(" If the step was not done")[0],
                "reference": " | ".join(r["target"]) or "none (done as written)",
                "note": ev.get("modified_description") or "",
                "answer": r["predicted"],
                "human": "",
            }
        )
        key.append(
            {"n": i, "judge": r["outcome"], "report_id": r["report_id"], "field_id": r["field_id"]}
        )
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    Path(args.out).with_suffix(".key.json").write_text(json.dumps(key, indent=1))
    print(
        f"{len(sheet)} rows ({len([s for s in sheet if s['reference'].startswith('none')])} with a none reference)"
    )
    print(f"label {args.out}; keep {Path(args.out).with_suffix('.key.json')} for compare")


def compare(args: argparse.Namespace) -> None:
    key = {k["n"]: k for k in json.loads(Path(args.key).read_text())}
    with open(args.sheet, newline="") as f:
        rows = list(csv.DictReader(f))
    todo = [r["n"] for r in rows if r["human"].strip() not in ("correct", "incorrect")]
    if todo:
        sys.exit(f"rows without a valid label (correct / incorrect): {todo[:10]}")
    agree = 0
    both = {
        ("correct", "correct"): 0,
        ("correct", "incorrect"): 0,
        ("incorrect", "correct"): 0,
        ("incorrect", "incorrect"): 0,
    }
    bad = []
    for r in rows:
        judge, human = key[int(r["n"])]["judge"], r["human"].strip()
        both[human, judge] += 1
        agree += judge == human
        if judge != human:
            bad.append((r, judge))
    n = len(rows)
    po = agree / n
    p_h = sum(v for (h, _), v in both.items() if h == "correct") / n
    p_j = sum(v for (_, j), v in both.items() if j == "correct") / n
    pe = p_h * p_j + (1 - p_h) * (1 - p_j)
    print(
        f"agreement {agree}/{n} = {po:.1%}   Cohen's kappa {(po - pe) / (1 - pe) if pe < 1 else 1.0:.2f}"
    )
    print(f"human correct / judge incorrect: {both['correct', 'incorrect']}   "
          f"human incorrect / judge correct: {both['incorrect', 'correct']}")  # fmt: skip
    for r, judge in bad:
        print(
            f"\n#{r['n']} judge={judge} human={r['human']}\n  reference: {r['reference']}\n  answer: {r['answer']}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(required=True)
    s = sub.add_parser("sample")
    s.add_argument("--run", required=True)
    s.add_argument("--eval", required=True, help="evaluation directory of that run")
    s.add_argument("--ground-truth", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("-n", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=sample)
    c = sub.add_parser("compare")
    c.add_argument("--sheet", required=True)
    c.add_argument("--key", required=True)
    c.set_defaults(func=compare)
    a = ap.parse_args()
    a.func(a)
