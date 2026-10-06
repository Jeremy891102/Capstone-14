"""Auto-build dense HD-EPIC reports: per video, the WINDOW-second window containing the most VQA items.
Usage: python scripts/build_hd_epic_dense.py <dir with vqa-benchmark/*.json> data/hd_epic_dense_v1 \
         [--videos N] [--window 300] [--max-q 50] [--video-ids ID ...]"""
import argparse, json, pathlib, re
ap = argparse.ArgumentParser()
ap.add_argument("vqa"); ap.add_argument("out")
ap.add_argument("--videos", type=int, default=3); ap.add_argument("--window", type=float, default=300)
ap.add_argument("--max-q", type=int, default=50)
ap.add_argument("--target-q", type=int, default=0, help="shortest window holding exactly this many questions (ignores --window)")
ap.add_argument("--max-window", type=float, default=0, help="with --target-q: skip videos needing a longer window"); ap.add_argument("--video-ids", nargs="*")
args = ap.parse_args()
V, OUT = pathlib.Path(args.vqa), pathlib.Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
CATS = {"fine_grained_action_localization": "loc", "fine_grained_how_recognition": "how",
        "fine_grained_why_recognition": "why", "ingredient_ingredient_weight": "weight",
        "recipe_step_recognition": "step"}
def sec(t): h, m, s = t.split(":"); return int(h)*3600 + int(m)*60 + float(s)
def rel(t, a): x = sec(t) - a; return f"{int(x//3600)}:{int(x%3600//60):02d}:{x%60:06.3f}"
def rel_t(x, a): x -= a; return f"{int(x//3600)}:{int(x%3600//60):02d}:{x%60:06.3f}"
def clip_text(s, a):  # original-timeline times -> clip-relative; "video 1" -> "this clip"
    s = re.sub(r"<TIME (\S+) video 1>", lambda m: rel(m[1], a), s)
    return s.replace("in the video 1", "in this clip").replace("in video 1", "in this clip").replace("video 1", "this clip")

items = {}  # video id -> [(start, end, category, qid, q)]
for cat in CATS:
    for qid, q in json.load(open(V / f"{cat}.json")).items():
        ins = list(q["inputs"].values())
        if len(ins) == 1 and "start_time" in ins[0] and len(set(q["choices"])) == len(q["choices"]):  # drop items with duplicate choices
            i = ins[0]; items.setdefault(i["id"], []).append((sec(i["start_time"]), sec(i["end_time"]), cat, qid, q))

def best_window(L):  # start at an item start; most items fully inside [a, a+W]
    return max(((sum(1 for s, e, *_ in L if s >= a and e <= a + args.window), a, args.window) for a, *_ in L))

def shortest_window(L):  # start at an item start; window ends at the N-th smallest item end
    n, best = args.target_q, None
    for a, *_ in L:
        E = sorted(e for s, e, *_ in L if s >= a)
        if len(E) >= n and (best is None or E[n - 1] - a < best[2]):
            best = (n, a, E[n - 1] - a)
    return best if best and (not args.max_window or best[2] <= args.max_window) else None

pick = shortest_window if args.target_q else best_window
cands = sorted(((*w, vid) for vid, L in items.items() if (not args.video_ids or vid in args.video_ids)
                and (w := pick(L))), key=lambda c: (-c[0], c[2], c[3]))[: args.videos]
reports, gt = [], []
for n, (cnt, a, win, vid) in enumerate(cands, 1):
    b = a + win; rid = f"hdepic-dense-{n:03d}"
    inside = [x for x in items[vid] if x[0] >= a and x[1] <= b + 1e-6][: args.max_q]
    fields = []
    for s, e, cat, qid, q in inside:
        fid = f"{CATS[cat]}_{qid.rsplit('_', 1)[1]}"
        text = clip_text(q["question"], a)
        if not re.search(r"\d:\d\d:\d\d", text):  # question has no time of its own -> state the segment
            text += f" (segment: {rel_t(s, a)} - {rel_t(e, a)} of this clip)"
        fields.append({"id": fid, "question": text, "answer_type": "single_choice",
                       "choices": [clip_text(c, a) for c in q["choices"]]})
        gt.append({"report_id": rid, "field_id": fid, "target": chr(65 + q["correct_idx"]),
                   "target_text": fields[-1]["choices"][q["correct_idx"]],
                   "evidence": {"hd_epic_question_id": qid, "category": cat}})
    reports.append({"id": rid, "input": {"videos": [{"id": "head_cam", "uri": f"{vid[:3]}/{vid}.mp4",
        "mime_type": "video/mp4", "time_range": {"start": a, "end": b, "unit": "seconds", "reference": "video_start"}}],
        "context": {"task_name": f"Kitchen cooking session ({vid[:3]})"}}, "fields": fields,
        "source": {"dataset": "hd-epic", "split": "vqa-benchmark", "original_ids": {"video_id": vid},
                   "annotation_refs": [x[3] for x in inside]}, "metadata": {"created_by": "build_hd_epic_dense"}})
w = lambda n, rows: (OUT / n).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
w("reports.jsonl", reports); w("ground_truth.jsonl", gt)
(OUT / "manifest.json").write_text(json.dumps({"schema_version": "video_report.v1", "benchmark_id": OUT.name,
  "description": f"Dense HD-EPIC: {len(reports)} windows ({f"shortest holding {args.target_q} Qs, max {args.max_window:.0f}s" if args.target_q else f"{args.window:.0f}s"}), {len(gt)} single-choice questions.",
  "reports_file": "reports.jsonl", "ground_truth_file": "ground_truth.jsonl", "mock_responses_file": None,
  "video_root": None, "source_datasets": ["hd-epic"],
  "notes": "Question timestamps are relative to the start of each clip (= time_range.start of the report)."}, indent=2) + "\n")
print([(r["id"], r["input"]["videos"][0]["uri"], len(r["fields"])) for r in reports])
