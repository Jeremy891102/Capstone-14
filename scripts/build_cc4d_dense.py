"""CaptainCook4D annotations -> video_report.v1 benchmark with mixed answer types (Template A).

Per report (one recording): 6 report-level fields, 3 fields per recipe step (correct / error
types / duration), and 2 decoy steps from other recipes whose 3 fields all have target
"not_visible" (hallucination checks). Decoys also appear in the step option lists (skipped /
first / longest; for skipped_steps they are correct answers, since they were not done). Steps
are named by description, never by number, and step blocks and option lists are shuffled, so
neither field order nor option lists reveal the real order or which steps are decoys.

Usage: python scripts/build_cc4d_dense.py <cc4d annotation dir> data/cc4d_dense_v1 \
         [--recordings 12_51 12_17] [--seed 0]
The annotation dir holds error_annotations.json and complete_step_annotations.json.
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
NV = "not_visible"
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}
STOP = {"the", "and", "with", "into", "for", "about", "until", "from", "over", "onto", "each"}

recs = {r["recording_id"]: r for r in json.loads((A / "error_annotations.json").read_text())}
steps_file = json.loads((A / "complete_step_annotations.json").read_text())
names = {k: v["activity_name"] for k, v in steps_file.items()}  # recording id -> recipe name
activity_names = {v["activity_id"]: v["activity_name"] for v in steps_file.values()}


def clean(desc):  # "Rinse-Rinse a tomato" -> "Rinse a tomato"
    return desc.split("-", 1)[1].strip() if "-" in desc else desc.strip()


def words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOP}


def tags_of(step):
    return {e["tag"].removesuffix(" Error") for e in step.get("errors") or []}


# every step of every recipe, for decoys: activity id -> {step_id: description}
recipe_steps = {}
for r in recs.values():
    for s in r["step_annotations"]:
        recipe_steps.setdefault(r["activity_id"], {})[s["step_id"]] = clean(s["description"])


def pick_decoys(activity, rng):
    own = recipe_steps[activity]
    own_words = set().union(*map(words, own.values()))
    own_texts = {t.lower() for t in own.values()}
    pool = [
        (len(words(t) & own_words), sid, t, a)
        for a, steps in recipe_steps.items()
        if a != activity
        for sid, t in steps.items()
        if t.lower() not in own_texts
    ]
    obvious = rng.choice([p for p in pool if p[0] == 0])
    plausible = rng.choice(sorted(pool, reverse=True)[:10])  # most word overlap with this recipe
    return [("obvious", *obvious[1:]), ("plausible", *plausible[1:])]


def q(fid, question, answer_type, target, evidence, choices=None):
    field = {"id": fid, "question": question, "answer_type": answer_type, "allow_not_visible": True}
    if choices:
        field["choices"] = choices
    return field, {"field_id": fid, "target": target, "evidence": evidence}


def letters(idxs):
    return [choice_letter(i) for i in sorted(idxs)]


reports, gt = [], []
for rid in args.recordings or sorted(recs, key=lambda k: [int(x) for x in k.split("_")]):
    r = recs[rid]
    rng = random.Random(f"{args.seed}:{rid}")
    steps = sorted(r["step_annotations"], key=lambda s: s["step_id"])
    done = [s for s in steps if s["start_time"] >= 0]  # -1 = Missing Step or annotation gap
    if not done:
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
    # decoys also go into every step option list, or their absence there gives them away
    decoys = pick_decoys(r["activity_id"], rng)
    decoy_texts = [d for _, _, d, _ in decoys]

    skip_opts = [label[id(s)] for s in steps] + decoy_texts
    rng.shuffle(skip_opts)
    missing = [label[id(s)] for s in steps if "Missing Step" in tags_of(s)]
    done_opts = [label[id(s)] for s in done] + decoy_texts
    rng.shuffle(done_opts)

    def letter_of(opts, text):
        return letters([opts.index(text)])[0]

    first = min(done, key=lambda s: s["start_time"])
    report_level = [
        q(
            "skipped_steps",
            "Which of these steps were not done in the video?",
            "multi_choice",
            letters(skip_opts.index(t) for t in missing + decoy_texts),
            {"skipped": missing, "decoys": decoy_texts},
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
            "num_steps_with_errors",
            "How many recipe steps were done wrong or skipped?",
            "int",
            sum(bool(tags_of(s)) for s in steps),
            {},
        ),
        q(
            "first_step",
            "Which of these steps was done first?",
            "single_choice",
            letter_of(done_opts, label[id(first)]),
            {"step": label[id(first)]},
            done_opts,
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
    if len(durs) < 2 or durs[0][0] >= 1.2 * durs[1][0]:  # skip when the top two are within 20%
        report_level.append(
            q(
                "longest_step",
                "Which of these steps took the longest?",
                "single_choice",
                letter_of(done_opts, durs[0][1]),
                {"step": durs[0][1], "seconds": round(durs[0][0], 1)},
                done_opts,
            )
        )

    # field ids get their "stepNN_" prefix after the shuffle: step_ids would expose decoys
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
    for kind, sid, d, src in decoys:
        ev = {"decoy": kind, "step_id": sid, "source_activity": activity_names[src]}
        blocks.append(
            [
                q("correct", f'Was the step "{d}" done correctly?', "bool", NV, ev),
                q(
                    "error_types",
                    f'What mistakes were made in the step "{d}"?',
                    "multi_choice",
                    NV,
                    ev,
                    TAGS,
                ),
                q(
                    "duration",
                    f'How many seconds did the step "{d}" take?',
                    "seconds",
                    NV,
                    ev,
                ),
            ]
        )
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
                "context": {"task_name": names.get(rid, str(r["activity_id"]))},
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
                "decoys": [{"kind": k, "step_id": sid} for k, sid, _, _ in decoys],
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
            "fields of mixed answer types, incl. 2 decoy steps per report (target not_visible).",
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
