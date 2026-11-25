"""Oracle clips with a single-step prompt (for Gemini Argon; video settings as in the full runs).

    python scripts/build_cc4d_oracle_single.py --pad 5 --out data/cc4d_oracle_single_v1

Differences from scripts/build_cc4d_oracle.py (which reused the full-report prompt and context):
* context is only the task name, the step instruction, and the definitions of the five error
  types that a clip of one step can show (Preparation, Measurement, Technique, Timing,
  Temperature). The full recipe is not included.
* `stepNN_correct` is asked neutrally ("performed correctly in this clip?").
* `stepNN_error_types` asks which mistakes, if any, are visible, with the five visible types only.
  Order and Missing Step are dropped from the options because a clip of one step cannot show
  them; the error_types target is the original target with those two letters removed (letters
  A-E are unchanged; the original target is kept in evidence.original_target).
* `stepNN_correct` keeps the original target (which counts Order errors); analysis reports
  Order-only steps separately.
Report ids, clips (annotated span +/- pad) and the set of steps are identical to cc4d_oracle_v1.
Offline; never invokes a model.
"""

import argparse
import json
import re
from pathlib import Path

VISIBLE = [
    "Preparation",
    "Measurement",
    "Technique",
    "Timing",
    "Temperature",
]  # letters A-E, as before
DEFS = (
    "- Preparation: the participant uses soiled or wrong ingredients, or uses different tools.\n"
    "- Measurement: the participant uses wrongly measured ingredients.\n"
    "- Technique: the participant performs the required action incorrectly, leading to a wrong"
    " outcome than expected.\n"
    "- Timing: the participant performs the step in a shorter or longer duration than prescribed"
    " (e.g. microwaving for a different time than the prescribed 30 seconds).\n"
    "- Temperature: the participant sets a higher or lower power level in the microwave or on a"
    " stove than prescribed."
)


def step_text(question: str) -> str:
    m = re.search(r'the step "(.*)" done correctly\?$', question)
    if not m:
        raise ValueError(f"unexpected question: {question!r}")
    return m[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/cc4d_k32_v1/fwd"))
    ap.add_argument("--out", type=Path, default=Path("data/cc4d_oracle_single_v1"))
    ap.add_argument("--pad", type=float, default=5.0)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output already exists")
    gt = {}
    for line in (args.src / "ground_truth.jsonl").read_text().splitlines():
        g = json.loads(line)
        gt[g["report_id"], g["field_id"]] = g
    reports, truths = [], []
    for line in (args.src / "reports.jsonl").read_text().splitlines():
        r = json.loads(line)
        tr = r["input"]["videos"][0]["time_range"]
        for f in r["fields"]:
            m = re.fullmatch(r"(step\d+)_correct", f["id"])
            if not m:
                continue
            g = gt[r["id"], f["id"]]
            start, end = g["evidence"]["start_time"], g["evidence"]["end_time"]
            if start < 0:
                continue
            sid, step = m[1], step_text(f["question"])
            has_et = (r["id"], f"{sid}_error_types") in gt
            rid = f"{r['id']}__{sid}"
            fields = [
                {
                    "id": f"{sid}_correct",
                    "question": f'Was the step "{step}" performed correctly in this clip?',
                    "answer_type": "bool",
                    "allow_not_visible": f.get("allow_not_visible", False),
                },
                {
                    "id": f"{sid}_error_types",
                    "question": "Which of these kinds of mistakes, if any, are visible in how "
                    f'the step "{step}" was performed? '
                    "Answer [] if the step was performed correctly.",
                    "answer_type": "multi_choice",
                    "choices": VISIBLE,
                    "allow_not_visible": f.get("allow_not_visible", False),
                },
            ][: 2 if has_et else 1]
            video = {**r["input"]["videos"][0]}
            video["time_range"] = {
                **tr,
                "start": round(max(0.0, start - args.pad), 3),
                "end": round(min(tr["end"], end + args.pad), 3),
            }
            context = {
                "task_name": r["input"]["context"]["task_name"],
                "step": step,
                "error_definitions": DEFS,
            }
            reports.append(
                {
                    **{
                        k: v for k, v in r.items() if k not in ("id", "fields", "input", "metadata")
                    },
                    "id": rid,
                    "fields": fields,
                    "input": {**r["input"], "videos": [video], "context": context},
                    "metadata": {
                        **r.get("metadata", {}),
                        "oracle_pad": args.pad,
                        "source_report": r["id"],
                    },
                }
            )
            truths.append({**g, "report_id": rid})
            if not has_et:
                continue
            et = gt[r["id"], f"{sid}_error_types"]
            truths.append(
                {
                    **et,
                    "report_id": rid,
                    "target": [x for x in et["target"] if x in "ABCDE"],
                    "evidence": {**et["evidence"], "original_target": et["target"]},
                }
            )
    args.out.mkdir(parents=True)

    def wr(name: str, rows: list[dict]) -> None:
        (args.out / name).write_text("".join(json.dumps(x) + "\n" for x in rows))

    wr("reports.jsonl", reports)
    wr("ground_truth.jsonl", truths)
    man = json.loads((args.src / "manifest.json").read_text())
    man.update(
        benchmark_id="cc4d_oracle_single_v1",
        description="CC4D oracle clips with a single-step prompt: one report per done step, "
        f"span +/- {args.pad} s.",
        notes="see scripts/build_cc4d_oracle_single.py",
    )
    (args.out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    spans = [r["input"]["videos"][0]["time_range"] for r in reports]
    mean = sum(s["end"] - s["start"] for s in spans) / len(spans)
    print(f"{len(reports)} reports, {len(truths)} fields, mean clip {mean:.0f} s")


if __name__ == "__main__":
    main()
