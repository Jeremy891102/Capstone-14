#!/usr/bin/env python3
"""Freeze the proposal's E1–E6 inputs and validate them without calling a provider.

Run with the repository interpreter; outputs are deliberately separate from source data.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

import yaml

from video_report.datasets.base import load_ground_truth, load_reports, validate_targets
from video_report.prompting import PromptTemplates, render
from video_report.snapshot import build_snapshot, code_revision

SYSTEM = """You answer questions about the provided video.

Base your answers on the provided evidence.
Follow the answer type and allowed values specified for each field.

Return exactly one valid JSON object using the requested field IDs as keys.
Do not include markdown or additional keys.
"""
PILOT_IDS = ["hdepic-dense-014", "hdepic-dense-022"]
EXPECTED = {
    "E1": 3212,
    "E2": 584,
    "E3": 292,
    "E4": 146,
    "E5": 73,
    "E6_homogeneous": 24,
    "E6_mixed": 24,
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered(fields, seed, report_id, role):
    result = sorted(fields, key=lambda f: f.id)
    digest = hashlib.sha256(f"{seed}:{report_id}:{role}".encode()).digest()
    random.Random(int.from_bytes(digest, "big")).shuffle(result)
    return result


def cues(target, companion):
    """Conservative surface cues only; no target labels/evidence are read."""
    action = re.search(r"<([^>]+)>", target.question)
    text = (companion.question + " " + " ".join(companion.choices)).lower()
    reasons = []
    if action:
        value = action.group(1).lower()
        words = set(re.findall(r"[a-z]+", value)) - {"the", "a", "up", "with", "left", "right"}
        if value in text:
            reasons.append("exact_action_mention")
        if len(words & set(re.findall(r"[a-z]+", text))) >= 2:
            reasons.append("shared_action_words")
    for option in target.choices:
        times = re.findall(r"\d+:\d+:\d+\.\d+", option)
        if len(times) == 2 and all(t in companion.question for t in times):
            reasons.append("option_interval_in_companion_question")
    return sorted(set(reasons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--probe-videos", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = repo / "data/hd_epic_q40_v1"
    out = args.output.resolve()
    if out.exists():
        raise SystemExit("Output exists; refusing to change a frozen selection.")
    reports = load_reports(source / "reports.jsonl")
    targets = load_ground_truth(source / "ground_truth.jsonl")
    selection = {r.id: [f.id for f in r.fields] for r in reports}
    gt = validate_targets(reports, targets, selection)
    assert len(gt) == len(targets) == 3212
    assert len(reports) == 76
    assert len({f.id for r in reports for f in r.fields}) == 3212
    counts = collections.Counter(f.id.split("_")[0] for r in reports for f in r.fields)
    assert counts == {"loc": 2861, "how": 175, "why": 160, "weight": 16}
    assert all(
        f.answer_type == "single_choice" and len(f.choices) == 5 for r in reports for f in r.fields
    )
    inventory = []
    for r in reports:
        assert len(r.input.videos) == 1
        v = r.input.videos[0]
        path = (args.video_root / v.uri).resolve()
        assert path.is_file(), str(path)
        tr = v.time_range
        assert tr is not None and 0 <= tr.start < tr.end
        duration = None
        if args.probe_videos:
            import imageio_ffmpeg

            proc = subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-i", str(path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", proc.stderr)
            assert match, f"Cannot probe {path}"
            duration = int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])
            assert tr.end <= duration + 0.05, (r.id, tr.end, duration)
        for f in r.fields:
            for h, m, sec in re.findall(
                r"(\d+):(\d+):([\d.]+)", f.question + " " + " ".join(f.choices)
            ):
                assert int(h) * 3600 + int(m) * 60 + float(sec) <= tr.end - tr.start + 0.05, f.id
        inventory.append(
            {
                "report_id": r.id,
                "video": v.model_dump(mode="json"),
                "absolute_path": str(path),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "probed_duration_s": duration,
                "question_count": len(r.fields),
            }
        )
    out.mkdir(parents=True)
    byid = {r.id: r for r in reports}
    pools = {}
    mixes = {}
    screening = []
    exclusions = []
    for r in reports:
        local = ordered(
            [f for f in r.fields if f.id.startswith("loc_")], args.seed, r.id, "localization"
        )
        if len(local) >= 32:
            pools[r.id] = local[:32]
        else:
            exclusions.append(
                {
                    "report_id": r.id,
                    "experiment": "E2-E5",
                    "reason": "fewer_than_32_localization",
                    "count": len(local),
                }
            )
        how = ordered([f for f in r.fields if f.id.startswith("how_")], args.seed, r.id, "how")
        why = ordered([f for f in r.fields if f.id.startswith("why_")], args.seed, r.id, "why")
        if len(local) < 8 or len(how) < 2 or len(why) < 2:
            exclusions.append(
                {"report_id": r.id, "experiment": "E6", "reason": "requires_loc8_how2_why2"}
            )
            continue
        companions = how[:2] + why[:2]
        # Use question/option text only to reduce obvious direct cues, never GT.
        ranked = sorted(
            enumerate(local),
            key=lambda item: (sum(len(cues(item[1], c)) for c in companions), item[0]),
        )
        common = [f for _, f in ranked[:4]]
        rest = [f for f in local if f.id not in {x.id for x in common}][:4]
        homo = common + rest
        mixed = common + companions
        mixes[r.id] = {"homogeneous": homo, "mixed": mixed, "targets": common}
        for condition, fields in [("homogeneous", homo), ("mixed", mixed)]:
            for target in common:
                for companion in fields[4:]:
                    for reason in cues(target, companion):
                        screening.append(
                            {
                                "report_id": r.id,
                                "condition": condition,
                                "target_id": target.id,
                                "companion_id": companion.id,
                                "reason": reason,
                            }
                        )
    assert len(pools) == 73 and len(mixes) == 24
    base = yaml.safe_load(
        (repo / "experiments/exp006_smoke5_vertex_whole_report/config.yaml").read_text()
    )
    user = (repo / "experiments/exp006_smoke5_vertex_whole_report/prompts/user.txt").read_text()
    user_template = PromptTemplates(SYSTEM, user)
    plan_index = {}
    e6_manifest = {}
    samples = []
    request_hashes = {}
    for experiment in EXPECTED:
        if experiment == "E1":
            selected = reports
        elif experiment.startswith("E6"):
            condition = experiment.removeprefix("E6_")
            selected = [
                byid[rid].model_copy(update={"fields": m[condition]}) for rid, m in mixes.items()
            ]
        else:
            selected = [
                byid[rid].model_copy(update={"fields": fields}) for rid, fields in pools.items()
            ]
        data = out / "data" / experiment
        jsonl(data / "reports.jsonl", [r.model_dump(mode="json") for r in selected])
        selected_pairs = {(r.id, f.id) for r in selected for f in r.fields}
        jsonl(
            data / "ground_truth.jsonl",
            [
                t.model_dump(mode="json")
                for t in targets
                if (t.report_id, t.field_id) in selected_pairs
            ],
        )
        dump(
            data / "manifest.json",
            {
                "schema_version": "video_report.v1",
                "benchmark_id": f"hdepic-baseline-{experiment}",
                "source_datasets": ["HD-EPIC"],
            },
        )
        configdir = out / "experiments" / experiment
        configdir.mkdir(parents=True)
        (configdir / "prompts").mkdir()
        (configdir / "prompts/system.txt").write_text(SYSTEM)
        (configdir / "prompts/user.txt").write_text(user)
        cfg = json.loads(json.dumps(base))
        cfg.update(
            name=f"baseline_20261017_{experiment}", description="Frozen proposal E1-E6; one order"
        )
        cfg["data"] = {
            "benchmark": str(data),
            "require_local_videos": True,
            "video_root": str(args.video_root.resolve()),
        }
        cfg["method"] = (
            {"name": "per_field"}
            if experiment == "E1"
            else (
                {
                    "name": "batched",
                    "batch_size": {"E2": 4, "E3": 8, "E4": 16, "E5": 32}[experiment],
                }
                if not experiment.startswith("E6")
                else {"name": "whole_report"}
            )
        )
        cfg["pricing"]["source"] = cfg["pricing"]["source"].replace("2026-10-08", "2026-10-17")
        (configdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        snapshot = build_snapshot(configdir / "config.yaml")
        assert len(snapshot.requests) == EXPECTED[experiment]
        # Reconstruct every prompt solely from allowed data; privileged records cannot enter it.
        for request in snapshot.requests:
            r = next(r for r in selected if r.id == request.report_id)
            fields = [r.field(fid) for fid in request.field_ids]
            assert (request.system, request.user) == render(
                user_template, fields, r.input.context, ["task_name"]
            )
            assert "ground_truth" not in request.user and "target_text" not in request.user
            assert request.system == SYSTEM
            assert request.videos[0].location.startswith(str(args.video_root.resolve()) + "/")
        jsonl(out / "plans" / f"{experiment}.jsonl", [p.to_json() for p in snapshot.plan])
        jsonl(out / "requests" / f"{experiment}.jsonl", [q.to_json() for q in snapshot.requests])
        request_hashes[experiment] = sha(out / "requests" / f"{experiment}.jsonl")
        plan_index[experiment] = [p.to_json() for p in snapshot.plan]
        pilot_requests = [q for q in snapshot.requests if q.report_id in PILOT_IDS]
        samples.extend(
            {"experiment": experiment, "request": q.to_json()}
            for q in pilot_requests
            if q == next(x for x in pilot_requests if x.report_id == q.report_id)
        )
        # All pilot settings are frozen now, but nothing is executed by this program.
        pcfg = json.loads(json.dumps(cfg))
        pcfg["data"]["report_ids"] = PILOT_IDS
        pcfg["name"] += "_pilot"
        (configdir / "pilot.yaml").write_text(yaml.safe_dump(pcfg, sort_keys=False))
        if experiment.startswith("E6"):
            for r in selected:
                e6_manifest.setdefault(r.id, {})[experiment] = [f.id for f in r.fields]
    for rid, fields in pools.items():
        desired = [f.id for f in fields]
        for e, size in [("E2", 4), ("E3", 8), ("E4", 16), ("E5", 32)]:
            batches = [p for p in plan_index[e] if p["report_id"] == rid]
            assert all(len(p["field_ids"]) == size for p in batches)
            assert [fid for p in batches for fid in p["field_ids"]] == desired
    for rid in mixes:
        h = e6_manifest[rid]["E6_homogeneous"]
        mixed = e6_manifest[rid]["E6_mixed"]
        assert len(set(h) & set(mixed)) == 4 and h[:4] == mixed[:4]
        e6_manifest[rid]["scored_target_ids"] = h[:4]
        e6_manifest[rid]["target_positions_1_based"] = [1, 2, 3, 4]
    jsonl(out / "rendered_request_samples.jsonl", samples)
    dump(
        out / "selection_manifest.json",
        {
            "seed": args.seed,
            "selection_algorithm": (
                "SHA256(seed:report:role) -> Python Random shuffle of ID-sorted fields"
            ),
            "e6_target_algorithm": (
                "rank seeded localization by surface cue count vs fixed H2/W2; "
                "choose lowest 4, stable ties"
            ),
            "source_hashes": {p.name: sha(p) for p in source.glob("*") if p.is_file()},
            "code": code_revision(),
            "preparation_script_sha256": sha(Path(__file__)),
            "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "localization32": {rid: [f.id for f in fs] for rid, fs in pools.items()},
            "e6": e6_manifest,
            "request_file_hashes": request_hashes,
            "exclusions": exclusions,
        },
    )
    dump(
        out / "vm_data_audit.json",
        {
            "reports": len(reports),
            "questions": len(gt),
            "subtypes": dict(counts),
            "localization32_eligible": len(pools),
            "mixing_eligible": len(mixes),
            "video_inventory": inventory,
            "source_data_unchanged": True,
        },
    )
    dump(out / "leakage_flags.json", screening)
    review = [
        "# E6 question and option review",
        "Surface screening is not proof of absence of semantic leakage.",
        "Target choices are printed without correct-answer labels. No GT is used for selection.",
    ]
    for rid, m in mixes.items():
        review += [f"\n## {rid}", "Targets: " + ", ".join(f.id for f in m["targets"])]
        for condition in ["homogeneous", "mixed"]:
            _, rendered = render(
                user_template, m[condition], byid[rid].input.context, ["task_name"]
            )
            review += [f"### {condition}", rendered]
        review += [json.dumps([x for x in screening if x["report_id"] == rid])]
    (out / "e6_question_review.md").write_text("\n\n".join(review))
    estimates = {}
    for e, plan in plan_index.items():
        video_seconds = sum(
            byid[p["report_id"]].input.videos[0].time_range.end
            - byid[p["report_id"]].input.videos[0].time_range.start
            for p in plan
        )
        nq = sum(len(p["field_ids"]) for p in plan)
        it = video_seconds * 64 + len(plan) * 114 + nq * 254.375
        ot = len(plan) * 1.2857142857 + nq * 13.1142857143
        estimates[e] = {
            "calls": len(plan),
            "answers": nq,
            "input_tokens": round(it),
            "output_tokens": round(ot),
            "estimated_cost_usd": (it * 0.3 + ot * 2.5) / 1e6,
        }
    pilot_counts = {
        e: sum(p["report_id"] in PILOT_IDS for p in plan) for e, plan in plan_index.items()
    }
    dump(
        out / "cost_estimate.json",
        {
            "basis": (
                "VM smoke observed ~64 video tokens/s; linear text/output fit; "
                "no cache discount; estimates only"
            ),
            "experiments": estimates,
            "total_calls": sum(EXPECTED.values()),
            "total_estimated_usd": sum(v["estimated_cost_usd"] for v in estimates.values()),
            "pilot_calls": pilot_counts,
            "pilot_total_calls": sum(pilot_counts.values()),
        },
    )
    dump(
        out / "offline_validation.json",
        {
            "status": "passed",
            "calls": EXPECTED,
            "source_gt_pairs_valid": len(gt),
            "identical_localization_pool_and_order": True,
            "e6_four_common_targets_positions_match": True,
            "allowed_input_only_rendering": True,
            "ground_truth_excluded_from_model_input_construction": True,
            "surface_cue_flags": len(screening),
            "semantic_leakage_absence_proven": False,
            "pilot_ids": PILOT_IDS,
            "pilot_calls": pilot_counts,
        },
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "counts": EXPECTED,
                "pilot_calls": pilot_counts,
                "output": str(out),
                "surface_cue_flags": len(screening),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
