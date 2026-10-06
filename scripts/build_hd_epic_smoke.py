"""Pick hand-chosen HD-EPIC VQA items -> video_report.v1 benchmark (manual smoke set).
Usage: python scripts/build_hd_epic_smoke.py <dir with vqa-benchmark/*.json> data/hd_epic_smoke_v1"""
import json, re, pathlib, sys
V = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
CAT = {"recipe_step_recognition": "step", "fine_grained_how_recognition": "how", "ingredient_ingredient_weight": "weight", "fine_grained_why_recognition": "why"}
# report id -> (video id, window start s, window end s, task context, [(file, qid)])
PICKS = {
 "hdepic-smoke-001": ("P07-20240529-134410", 230, 410, "Chopped chickpea salad (P07)",
    [("recipe_step_recognition", "recipe_step_recognition_93"), ("ingredient_ingredient_weight", "ingredient_ingredient_weight_43"),
     ("recipe_step_recognition", "recipe_step_recognition_22"), ("ingredient_ingredient_weight", "ingredient_ingredient_weight_44")]),
 "hdepic-smoke-002": ("P07-20240529-194518", 1555, 1680, "Mushroom risotto (P07)",
    [("recipe_step_recognition", "recipe_step_recognition_32"), ("fine_grained_why_recognition", "fine_grained_why_recognition_225"),
     ("recipe_step_recognition", "recipe_step_recognition_43")]),
 "hdepic-smoke-003": ("P09-20240623-120359", 80, 180, "Cooking session (P09)",
    [("fine_grained_how_recognition", "fine_grained_how_recognition_424"), ("fine_grained_how_recognition", "fine_grained_how_recognition_242"),
     ("fine_grained_how_recognition", "fine_grained_how_recognition_365"), ("ingredient_ingredient_weight", "ingredient_ingredient_weight_7")]),
}
def ts(t): h, m, s = t.split(":"); return f"{int(h)}:{m}:{s}"
def sec(t): h, m, s = t.split(":"); return int(h)*3600 + int(m)*60 + float(s)
def rel(t, a):  # original-timeline timestamp -> clip-relative H:MM:SS.mmm (clips start at 0 s)
    x = sec(t) - a; return f"{int(x//3600)}:{int(x%3600//60):02d}:{x%60:06.3f}"
reports, gt = [], []
for rid, (vid, a, b, task, picks) in PICKS.items():
    fields = []
    for f, qid in picks:
        q = json.load(open(V / f"{f}.json"))[qid]; inp = q["inputs"]["video 1"]
        text = re.sub(r"<TIME (\S+) video 1>", lambda m: rel(m[1], a), q["question"]).replace("in video 1", "in this video")
        text = text.replace("in this video", "in this clip")
        if "start_time" in inp and not re.search(r"\d\d:\d\d:\d\d", text):
            text += f" (segment: {rel(inp['start_time'], a)} - {rel(inp['end_time'], a)} of this clip)"
        # sanity: question segment must fall inside the report window
        if "start_time" in inp: assert a <= sec(inp["start_time"]) and sec(inp["end_time"]) <= b, qid
        fid = f"{CAT[f]}_{qid.rsplit('_',1)[1]}"
        fields.append({"id": fid, "question": text, "answer_type": "single_choice", "choices": q["choices"]})
        L = chr(65 + q["correct_idx"])
        gt.append({"report_id": rid, "field_id": fid, "target": L, "target_text": q["choices"][q["correct_idx"]],
                   "evidence": {"hd_epic_question_id": qid, "category": f}})
    reports.append({"id": rid,
      "input": {"videos": [{"id": "head_cam", "uri": f"{vid[:3]}/{vid}.mp4", "mime_type": "video/mp4",
                            "time_range": {"start": float(a), "end": float(b), "unit": "seconds", "reference": "video_start"}}],
                "context": {"task_name": task}},
      "fields": fields,
      "source": {"dataset": "hd-epic", "split": "vqa-benchmark", "original_ids": {"video_id": vid}, "annotation_refs": [q for _, q in picks]},
      "metadata": {"created_by": "manual-smoke-picks"}})
w = lambda n, rows: (OUT / n).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
w("reports.jsonl", reports); w("ground_truth.jsonl", gt)
(OUT / "manifest.json").write_text(json.dumps({"schema_version": "video_report.v1", "benchmark_id": "hd_epic_smoke_v1",
  "description": "Manual HD-EPIC smoke set: 3 videos, 11 single-choice questions.", "reports_file": "reports.jsonl",
  "ground_truth_file": "ground_truth.jsonl", "mock_responses_file": None, "video_root": None,
  "source_datasets": ["hd-epic"], "notes": "Question timestamps are relative to the start of each clip (= time_range.start of the report)."}, indent=2) + "\n")
