"""Build a draft single-type nested pilot. Offline; never invokes a model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from video_report.benchmarks.localization_pilot import check_pairs, nested_selection
from video_report.datasets.base import load_benchmark, load_ground_truth, load_reports


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("data/hd_epic_q40_v1"))
    ap.add_argument("--out", type=Path, default=Path("data/hd_epic_loc_nested_v1"))
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--videos", type=int, default=3)
    args = ap.parse_args()
    if args.videos <= 0:
        ap.error("--videos must be positive")
    if args.out.exists():
        ap.error("output already exists; choose a new directory to preserve review work")
    benchmark = load_benchmark(args.source.resolve())
    reports = [r.model_dump(mode="json") for r in load_reports(benchmark.reports_path)]
    gt = [g.model_dump(mode="json") for g in load_ground_truth(benchmark.ground_truth_path)]
    targetmap = {(g["report_id"], g["field_id"]): g for g in gt}
    chosen = []
    participants = set()
    # Deterministic source order; prefer different participants for the pilot.
    for report in sorted(reports, key=lambda r: r["id"]):
        participant = report["input"]["videos"][0]["uri"].split("/")[0]
        if participant in participants:
            continue
        targets = {f["id"]: targetmap[(report["id"], f["id"])] for f in report["fields"]}
        selection = nested_selection(report, targets, seed=args.seed)
        if selection is None:
            continue
        check_pairs(selection)
        chosen.append(selection)
        participants.add(participant)
        if len(chosen) == args.videos:
            break
    if len(chosen) < args.videos:
        raise SystemExit(
            f"Only {len(chosen)} participants meet the 32-question constraints; no output written"
        )
    args.out.mkdir(parents=True)
    review = [
        "# 定位題巢狀 Pilot 人工核對稿",
        "",
        "包含標準答案，只供資料檢查；不能送給模型。尚未核對原始影片與語意提示。",
        "",
    ]
    summary = []
    for selection in chosen:
        r = selection["report"]
        largest = selection["subsets"][32]
        review += [f"## {r['id']} — {r['input']['videos'][0]['uri']}", ""]
        for f in sorted(largest, key=lambda f: selection["intervals"][f["id"]][0]):
            sizes = [n for n in [8, 16, 32] if f in selection["subsets"][n]]
            review += [f"### {f['id']} · sizes={sizes}", "", f["question"], ""]
            review += [f"- {chr(65 + i)}. {c}" for i, c in enumerate(f["choices"])]
            g = targetmap[(r["id"], f["id"])]
            review += [
                "",
                f"答案：{g['target']}；證據：{selection['intervals'][f['id']]}",
                "",
                "- [ ] 核對影片及時間映射",
                "- [ ] 檢查互相提示與短動作取樣",
                "",
            ]
        summary.append(
            {
                "report_id": r["id"],
                "video": r["input"]["videos"][0],
                "counts": {n: len(fs) for n, fs in selection["subsets"].items()},
                "time_thirds": selection["time_thirds"],
            }
        )
    for size in [8, 16, 32]:
        for order in range(3):
            name = f"loc_n{size:02d}_order{order + 1}"
            rows = []
            for selection in chosen:
                row = copy.deepcopy(selection["report"])
                byid = {f["id"]: f for f in selection["subsets"][size]}
                row["fields"] = [byid[q] for q in selection["orders"][order] if q in byid]
                row["metadata"].update(
                    {
                        "pilot_seed": args.seed,
                        "size": size,
                        "order": order + 1,
                        "review_status": "draft_video_and_semantic_review_pending",
                    }
                )
                rows.append(row)
            write_set(args.out, name, rows, targetmap)
    rows = []
    for selection in chosen:
        row = copy.deepcopy(selection["report"])
        row["fields"] = selection["subsets"][32]
        row["metadata"].update({"review_status": "draft_video_and_semantic_review_pending"})
        rows.append(row)
    write_set(args.out, "loc_per_field", rows, targetmap)
    (args.out / "question_review.md").write_text("\n".join(review))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    files = ["reports.jsonl", "ground_truth.jsonl", "manifest.json"]
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "source": str(args.source),
                "source_sha256": {
                    n: hashlib.sha256((args.source / n).read_bytes()).hexdigest() for n in files
                },
                "api_calls": 0,
                "status": "draft_video_and_semantic_review_pending",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def write_set(out, name, rows, targetmap):
    root = out / name
    root.mkdir()
    write_jsonl(root / "reports.jsonl", rows)
    write_jsonl(
        root / "ground_truth.jsonl",
        [targetmap[(r["id"], f["id"])] for r in rows for f in r["fields"]],
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "video_report.v1",
                "benchmark_id": name,
                "reports_file": "reports.jsonl",
                "ground_truth_file": "ground_truth.jsonl",
                "source_datasets": ["hd-epic"],
                "notes": "Localization: fixed clip, nested sizes, matched orders. Draft only.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
