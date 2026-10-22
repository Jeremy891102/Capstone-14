"""Build hd_epic_q2_full_v1: every how/why/weight VQA item on the 76 hd_epic_q40_v1 videos.
Items already in q40_v1 keep their window and exact text (E1 predictions stay reusable).
New items get windows of the same length as that video's q40_v1 window, at a random offset.
Usage: python scripts/build_hd_epic_q2_full.py <dir with vqa-benchmark/*.json> \
         data/hd_epic_q40_v1 data/hd_epic_q2_full_v1 --durations video_durations.json"""

import argparse
import json
import pathlib
import random
import re

ap = argparse.ArgumentParser()
ap.add_argument("vqa")
ap.add_argument("q40")
ap.add_argument("out")
ap.add_argument("--durations", required=True, help="JSON {video_id: seconds}")
ap.add_argument("--seed", type=int, default=20261022)
args = ap.parse_args()
V, Q40, OUT = map(pathlib.Path, (args.vqa, args.q40, args.out))
OUT.mkdir(parents=True, exist_ok=True)
CATS = {
    "fine_grained_how_recognition": "how",
    "fine_grained_why_recognition": "why",
    "ingredient_ingredient_weight": "weight",
}
dur = json.loads(pathlib.Path(args.durations).read_text())
rng = random.Random(args.seed)


def sec(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def rel_t(x, a):
    x -= a
    return f"{int(x // 3600)}:{int(x % 3600 // 60):02d}:{x % 60:06.3f}"


def clip_text(s, a):  # same rewrite as build_hd_epic_dense.py
    s = re.sub(r"<TIME (\S+) video 1>", lambda m: rel_t(sec(m[1]), a), s)
    return (
        s.replace("in the video 1", "in this clip")
        .replace("in video 1", "in this clip")
        .replace("video 1", "this clip")
    )


def jl(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


# 1) q40_v1 reports, restricted to how/why/weight fields (verbatim)
reports, gt, done = [], [], set()
q40_gt = jl(Q40 / "ground_truth.jsonl")
windows = {}  # video id -> q40 window length
for r in jl(Q40 / "reports.jsonl"):
    v = r["input"]["videos"][0]
    vid = pathlib.Path(v["uri"]).stem
    windows[vid] = v["time_range"]["end"] - v["time_range"]["start"]
    keep = {f["id"] for f in r["fields"] if f["id"].split("_")[0] in CATS.values()}
    rows = [t for t in q40_gt if t["report_id"] == r["id"] and t["field_id"] in keep]
    done |= {t["evidence"]["hd_epic_question_id"] for t in rows}
    if keep:
        reports.append({**r, "fields": [f for f in r["fields"] if f["id"] in keep]})
        reports[-1]["source"]["annotation_refs"] = [
            t["evidence"]["hd_epic_question_id"] for t in rows
        ]
        gt += rows
assert len(windows) == 76 and windows.keys() == dur.keys()

# 2) remaining items on the same videos (same filters as build_hd_epic_dense.py)
items = {}  # video id -> [(start, end, cat, qid, q)]
for cat in CATS:
    for qid, q in json.loads((V / f"{cat}.json").read_text()).items():
        ins = list(q["inputs"].values())
        if (
            len(ins) == 1
            and "start_time" in ins[0]
            and ins[0]["id"] in windows
            and qid not in done
            and len(set(q["choices"])) == len(q["choices"])
        ):
            i = ins[0]
            items.setdefault(i["id"], []).append(
                (sec(i["start_time"]), sec(i["end_time"]), cat, qid, q)
            )

n = 0
for vid in sorted(items):
    W, todo = windows[vid], sorted(items[vid], key=lambda x: x[:2])
    while todo:  # greedy: random window around the earliest uncovered item, take all items inside
        s, e = todo[0][:2]
        assert e - s <= W and e <= dur[vid] + 0.05, (vid, todo[0][3])
        a = min(max(rng.uniform(max(0, e - W), s), 0), max(dur[vid] - W, 0))
        b = min(a + W, dur[vid])
        inside = [x for x in todo if x[0] >= a and x[1] <= b + 1e-6]
        todo = [x for x in todo if x not in inside]
        n += 1
        rid = f"hdepic-q2-{n:03d}"
        fields = []
        for s, e, cat, qid, q in inside:
            fid = f"{CATS[cat]}_{qid.rsplit('_', 1)[1]}"
            text = clip_text(q["question"], a)
            if not re.search(r"\d:\d\d:\d\d", text):
                text += f" (segment: {rel_t(s, a)} - {rel_t(e, a)} of this clip)"
            fields.append(
                {
                    "id": fid,
                    "question": text,
                    "answer_type": "single_choice",
                    "choices": [clip_text(c, a) for c in q["choices"]],
                }
            )
            gt.append(
                {
                    "report_id": rid,
                    "field_id": fid,
                    "target": chr(65 + q["correct_idx"]),
                    "target_text": fields[-1]["choices"][q["correct_idx"]],
                    "evidence": {"hd_epic_question_id": qid, "category": cat},
                }
            )
        reports.append(
            {
                "id": rid,
                "input": {
                    "videos": [
                        {
                            "id": "head_cam",
                            "uri": f"{vid[:3]}/{vid}.mp4",
                            "mime_type": "video/mp4",
                            "time_range": {
                                "start": a,
                                "end": b,
                                "unit": "seconds",
                                "reference": "video_start",
                            },
                        }
                    ],
                    "context": {"task_name": f"Kitchen cooking session ({vid[:3]})"},
                },
                "fields": fields,
                "source": {
                    "dataset": "hd-epic",
                    "split": "vqa-benchmark",
                    "original_ids": {"video_id": vid},
                    "annotation_refs": [x[3] for x in inside],
                },
                "metadata": {"created_by": "build_hd_epic_q2_full"},
            }
        )

# every field id unique, every question's times inside its clip
assert len({f["id"] for r in reports for f in r["fields"]}) == len(gt)
for r in reports:
    tr = r["input"]["videos"][0]["time_range"]
    for f in r["fields"]:
        for t in re.findall(r"\d+:\d\d:[\d.]+", f["question"] + " " + " ".join(f["choices"])):
            assert sec(t) <= tr["end"] - tr["start"] + 0.05, (r["id"], f["id"])


def w(name, rows):
    (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


w("reports.jsonl", reports)
w("ground_truth.jsonl", gt)
counts = {c: sum(t["field_id"].startswith(c + "_") for t in gt) for c in CATS.values()}
(OUT / "manifest.json").write_text(
    json.dumps(
        {
            "schema_version": "video_report.v1",
            "benchmark_id": OUT.name,
            "description": f"HD-EPIC Q2 subtypes on the 76 q40_v1 videos: {len(reports)} windows, "
            f"{len(gt)} single-choice questions {counts}.",
            "reports_file": "reports.jsonl",
            "ground_truth_file": "ground_truth.jsonl",
            "mock_responses_file": None,
            "video_root": None,
            "source_datasets": ["hd-epic"],
            "notes": "Question timestamps are relative to the start of each clip "
            "(= time_range.start of the report). hdepic-dense-* reports are q40_v1 verbatim.",
        },
        indent=2,
    )
    + "\n"
)
print(len(reports), "reports", counts)
