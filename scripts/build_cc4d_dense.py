"""CaptainCook4D annotations -> video_report.v1 benchmark with mixed answer types (Template A).

Per report (one recording): 4-5 report-level fields and 3 fields per recipe step (correct / error
types / duration). The skipped_steps and longest_step option lists have exactly 8 options (real
steps only), so recipe length does not change the number of options; recordings with fewer than 8
filmed steps are dropped. Steps are named by description, never by number, and step blocks and
option lists are shuffled, so neither field order nor option lists reveal the real step order.

Usage: python scripts/build_cc4d_dense.py <cc4d annotation dir> data/cc4d_dense_v1 \
         [--recordings 12_51 12_17] [--seed 0]
The annotation dir holds error_annotations.json, complete_step_annotations.json and task_graphs/
(the 24 <recipe>.json files from github.com/CaptainCook4D/annotations).

Each report's input.context carries `task_name`, `error_definitions` (the 7 error categories, from
the CaptainCook4D paper) and `recipe`: the recipe's steps with the steps each
must come after, taken from the task graph (prerequisite order, not a single fixed sequence).
"""

import argparse
import json
import pathlib
import random
import re

from video_report.benchmarks.schema import choice_letter

ap = argparse.ArgumentParser()
ap.add_argument("annotations")
ap.add_argument("out")
ap.add_argument("--recordings", nargs="*")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
A, OUT = pathlib.Path(args.annotations), pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

TAGS = ["Preparation", "Measurement", "Technique", "Timing", "Temperature", "Order", "Missing Step"]
# Error category definitions, from Section 3.2 of the CaptainCook4D paper (arXiv 2312.14556)
ERROR_DEFINITIONS = "\n".join(
    f"- {t}: {d}"
    for t, d in {
        "Preparation": "the participant uses soiled or wrong ingredients, or uses different tools.",
        "Measurement": "the participant uses wrongly measured ingredients.",
        "Technique": "the participant performs the required action incorrectly, leading to a wrong "
        "outcome than expected.",
        "Timing": "the participant performs a step in a shorter or longer duration than what is "
        "prescribed (e.g. microwaving for a different time than the prescribed 30 seconds).",
        "Temperature": "the participant sets a higher or lower power level in the microwave or on "
        "a stove than what is prescribed.",
        "Order": "the participant executes steps out of the required sequence.",
        "Missing Step": "the participant omits to perform a step.",
    }.items()
)
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}
N_REAL = 8  # step option lists: 8 real steps, whatever the recipe length

recs = {r["recording_id"]: r for r in json.loads((A / "error_annotations.json").read_text())}
steps_file = json.loads((A / "complete_step_annotations.json").read_text())
names = {k: v["activity_name"] for k, v in steps_file.items()}  # recording id -> recipe name
activity_names = {v["activity_id"]: v["activity_name"] for v in steps_file.values()}


def clean(desc):  # "Rinse-Rinse a tomato" -> "Rinse a tomato"
    return desc.split("-", 1)[1].strip() if "-" in desc else desc.strip()


def recipe_text(name):
    """Recipe steps and their prerequisites from the task graph, as model-facing text."""
    g = json.loads((A / "task_graphs" / (re.sub(r"[^a-z]", "", name.lower()) + ".json")).read_text())
    end = max(map(int, g["steps"]))
    desc = {int(k): clean(v) for k, v in g["steps"].items() if int(k) not in (0, end)}
    after = {i: [] for i in desc}
    for a, b in g["edges"]:
        if a in desc and b in desc:
            after[b].append(a)
    depth = {}

    def d(i):
        depth[i] = depth.get(i) or 1 + max(map(d, after[i]), default=0)
        return depth[i]

    for i in desc:
        d(i)
    label = dict(desc)
    for text in {t for t in desc.values() if list(desc.values()).count(t) > 1}:
        same = sorted((i for i in desc if desc[i] == text), key=lambda i: (depth[i], i))
        for n, i in enumerate(same, 1):
            label[i] = f"{text} ({ORDINAL[n]} time)"
    lines = [
        f"- {label[i]}" + (f" (after: {'; '.join(label[j] for j in sorted(after[i]))})" if after[i] else "")
        for i in sorted(desc)
    ]
    return list(label.values()), (
        "A step may only be done after every step in its 'after' list is finished; "
        "steps with no 'after' can be done at any point, and steps that do not depend on each "
        "other can be done in either order.\n" + "\n".join(lines)
    )


def tags_of(step):
    return {e["tag"].removesuffix(" Error") for e in step.get("errors") or []}


def q(fid, question, answer_type, target, evidence, choices=None):
    field = {"id": fid, "question": question, "answer_type": answer_type}
    if choices:
        field["choices"] = choices
    return field, {"field_id": fid, "target": target, "evidence": evidence}


def letters(idxs):
    return [choice_letter(i) for i in sorted(idxs)]


norm = lambda t: re.sub(r"\s+", " ", t.lower())
recipes = {a: recipe_text(n) for a, n in activity_names.items()}
recipe_labels = {a: {norm(t) for t in labels} for a, (labels, _) in recipes.items()}
reports, gt = [], []
for rid in args.recordings or sorted(recs, key=lambda k: [int(x) for x in k.split("_")]):
    r = recs[rid]
    rng = random.Random(f"{args.seed}:{rid}")
    steps = sorted(r["step_annotations"], key=lambda s: s["step_id"])
    done = [s for s in steps if s["start_time"] >= 0]  # -1 = Missing Step or annotation gap
    if len(done) < N_REAL:  # too few steps filmed to fill an 8-option list
        continue
    # model-facing names; a description that repeats gets "(1st time)", ... in time order
    label = {id(s): clean(s["description"]) for s in steps}
    for text in {t for t in label.values() if list(label.values()).count(t) > 1}:
        same = sorted(
            (s for s in steps if label[id(s)] == text),
            key=lambda s: (s["start_time"] < 0, s["start_time"]),
        )
        for n, s in enumerate(same, 1):
            label[id(s)] = f"{text} ({ORDINAL[n]} time)"
    # a step the recipe text does not list (a repeated step, or a variant wording) would confuse a
    # model that reads the recipe: drop the recording
    if any(norm(label[id(s)]) not in recipe_labels[r["activity_id"]] for s in steps):
        continue
    missing = [label[id(s)] for s in steps if "Missing Step" in tags_of(s)]
    others = [label[id(s)] for s in steps if label[id(s)] not in missing]
    skip_opts = missing + rng.sample(others, N_REAL - len(missing))
    rng.shuffle(skip_opts)

    def longest_opts(answer):  # the answer and 7 other filmed steps, shuffled
        rest = [label[id(s)] for s in done if label[id(s)] != answer]
        opts = [answer] + rng.sample(rest, N_REAL - 1)
        rng.shuffle(opts)
        return opts

    def letter_of(opts, text):
        return letters([opts.index(text)])[0]

    report_level = [
        q(
            "skipped_steps",
            "Which of these steps were not done in the video?",
            "multi_choice",
            letters(skip_opts.index(t) for t in missing),
            {"skipped": missing},
            skip_opts,
        ),
        q(
            "order_violated",
            "Were any steps done out of the order the recipe requires?",
            "bool",
            any("Order" in tags_of(s) for s in steps),
            {},
        ),
        q(
            "steps_done",
            "How many of the recipe steps were done in the video?",
            "int",
            len(done),
            {},
        ),
        q(
            "total_duration",
            "How many seconds passed from the start of the first step to the end of the last step?",
            "seconds",
            round(max(s["end_time"] for s in done) - min(s["start_time"] for s in done), 1),
            {},
        ),
    ]
    durs = sorted(((s["end_time"] - s["start_time"], label[id(s)]) for s in done), reverse=True)
    if durs[0][0] >= 1.2 * durs[1][0]:  # skip when the top two are within 20%
        opts = longest_opts(durs[0][1])
        report_level.append(
            q(
                "longest_step",
                "Which of these steps took the longest?",
                "single_choice",
                letter_of(opts, durs[0][1]),
                {"step": durs[0][1], "seconds": round(durs[0][0], 1)},
                opts,
            )
        )

    # field ids get their "stepNN_" prefix after the shuffle: step_ids would expose the step order
    blocks = []
    for s in steps:
        d, t = label[id(s)], tags_of(s)
        ev = {
            "step_id": s["step_id"],
            "start_time": s["start_time"],
            "end_time": s["end_time"],
            "errors": s.get("errors") or [],
        }
        block = [q("correct", f'Was the step "{d}" done correctly?', "bool", not t, ev)]
        if "Other" not in t:  # 8 steps tagged "Other": not in the option list, so no target
            block.append(
                q(
                    "error_types",
                    f'What mistakes were made in the step "{d}"?',
                    "multi_choice",
                    letters(TAGS.index(x) for x in t),
                    ev,
                    TAGS,
                )
            )
        if s["start_time"] >= 0:
            block.append(
                q(
                    "duration",
                    f'How many seconds did the step "{d}" take?',
                    "seconds",
                    round(s["end_time"] - s["start_time"], 1),
                    ev,
                )
            )
        blocks.append(block)
    rng.shuffle(blocks)
    for n, block in enumerate(blocks, 1):
        for field, target in block:
            field["id"] = target["field_id"] = f"step{n:02d}_{field['id']}"
    pairs = report_level + [p for b in blocks for p in b]

    end = max(s["end_time"] for s in done) + 5.0  # clip ends shortly after the last step
    reports.append(
        {
            "id": f"cc4d-{rid}",
            "input": {
                "videos": [
                    {
                        "id": "gopro",
                        "uri": f"{rid}_360p.mp4",
                        "mime_type": "video/mp4",
                        "time_range": {
                            "start": 0.0,
                            "end": round(end, 3),
                            "unit": "seconds",
                            "reference": "video_start",
                        },
                    }
                ],
                "context": {
                    "task_name": names.get(rid, str(r["activity_id"])),
                    "recipe": recipes[r["activity_id"]][1],
                    "error_definitions": ERROR_DEFINITIONS,
                },
            },
            "fields": [f for f, _ in pairs],
            "source": {
                "dataset": "captaincook4d",
                "split": "all",
                "original_ids": {"recording_id": rid},
                "annotation_refs": [f"error_annotations.json#recording_id={rid}"],
            },
            "metadata": {
                "created_by": "build_cc4d_dense",
                "seed": args.seed,
                "num_steps": len(steps),
            },
        }
    )
    gt += [{"report_id": f"cc4d-{rid}", **t} for _, t in pairs]


def w(name, rows):
    (OUT / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))


w("reports.jsonl", reports)
w("ground_truth.jsonl", gt)
(OUT / "manifest.json").write_text(
    json.dumps(
        {
            "schema_version": "video_report.v1",
            "benchmark_id": OUT.name,
            "description": f"CaptainCook4D Template A: {len(reports)} recordings, {len(gt)} "
            "fields of mixed answer types.",
            "reports_file": "reports.jsonl",
            "ground_truth_file": "ground_truth.jsonl",
            "mock_responses_file": None,
            "video_root": None,
            "source_datasets": ["captaincook4d"],
            "notes": "Video uris are <recording_id>_360p.mp4 (GoPro 360p). time_range ends 5 s "
            "after the last annotated step.",
        },
        indent=2,
    )
    + "\n"
)
print(f"{len(reports)} reports, {len(gt)} fields")
