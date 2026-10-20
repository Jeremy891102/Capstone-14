#!/usr/bin/env python3
"""Audit saved pilot evidence and export paired statistics; never calls a provider."""

# Report paragraphs and command examples are kept intact for readability.
# ruff: noqa: E501
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import zipfile
from pathlib import Path

from prepare_hd_epic_baseline import PILOT_IDS, SYSTEM, dump, sha


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    root = args.freeze.resolve()
    pilot = root / "pilot"
    out = root / "report"
    out.mkdir(exist_ok=True)
    manifest = load(root / "selection_manifest.json")
    summaries = load(pilot / "pilot_results.json")
    assert len(summaries) == 7
    scores = {}
    all_calls = []
    logs = []
    costchecks = []
    cliphashes = {}
    for phase in summaries:
        e = phase["experiment"]
        run = Path(phase["run_dir"])
        ev = Path(phase["evaluation_dir"])
        scores[e] = rows(ev / "field_scores.jsonl")
        requests = {q["call_id"]: q for q in rows(run / "snapshot/requests.jsonl")}
        sessions = [load(p) for p in sorted((pilot / e).glob("timing_previous_*.json"))]
        sessions.append(phase["timing"])
        api = {q["call_id"]: q for t in sessions for q in t["api"]}
        media = {q["call_id"]: q for t in sessions for q in t["media"]}
        phase["timing"]["execution_wall_s"] = sum(t["execution_wall_s"] for t in sessions)
        logs.append(pilot / e / "smoke_test.log")
        raw_total = 0
        for p in sorted((run / "calls").glob("*.json")):
            call = load(p)
            assert call["status"] == "succeeded" and len(call["attempts"]) == 1
            a = call["attempts"][0]
            req = requests[call["call_id"]]
            assert req["system"] == SYSTEM and req["report_id"] in PILOT_IDS
            body = load(pilot / e / "wire_inputs" / p.name)
            assert body["systemInstruction"]["parts"] == [{"text": SYSTEM}]
            assert body["contents"][0]["parts"][-1]["text"] == req["user"]
            encoded = next(
                x["inlineData"]["data"] for x in body["contents"][0]["parts"] if "inlineData" in x
            )
            digest = hashlib.sha256(base64.b64decode(encoded, validate=True)).hexdigest()
            m = media[call["call_id"]]
            assert digest == m["clip_sha256"] == sha(Path(m["clip_path"]))
            previous = cliphashes.setdefault(call["report_id"], digest)
            assert previous == digest
            raw = load(pilot / e / "raw_outputs" / p.name)
            u = raw["usageMetadata"]
            assert (
                u["promptTokenCount"] + u["candidatesTokenCount"] + u.get("thoughtsTokenCount", 0)
                == u["totalTokenCount"]
            )
            assert sum(x["tokenCount"] for x in u["promptTokensDetails"]) == u["promptTokenCount"]
            assert any(
                x["modality"] == "VIDEO" and x["tokenCount"] > 0 for x in u["promptTokensDetails"]
            )
            cached = u.get("cachedContentTokenCount", 0)
            cost = (
                (u["promptTokenCount"] - cached) * 0.3
                + cached * 0.03
                + (u["candidatesTokenCount"] + u.get("thoughtsTokenCount", 0)) * 2.5
            ) / 1e6
            raw_total += cost
            response = load(run / a["response_file"])["response"]
            assert response["finish_reason"] == "STOP"
            assert set(json.loads(response["text"])) == set(call["field_ids"])
            all_calls.append(
                {
                    "experiment": e,
                    "call_id": call["call_id"],
                    "report_id": call["report_id"],
                    "field_ids": call["field_ids"],
                    "request": req,
                    "response": response,
                    "api_round_trip_s": api[call["call_id"]]["api_round_trip_s"],
                    "provider_latency_s": a["latency_s"],
                    "estimated_cost_usd": cost,
                    "clip_sha256": digest,
                    "usage": a["usage"],
                }
            )
        assert abs(raw_total - phase["metrics"]["usage"]["cost"]) < 1e-9
        costchecks.append(
            {
                "experiment": e,
                "raw_cost": raw_total,
                "evaluator_cost": phase["metrics"]["usage"]["cost"],
                "matches": True,
            }
        )
    assert len(all_calls) == 124
    shared = {(rid, fid) for rid in PILOT_IDS for fid in manifest["localization32"][rid]}
    assert len(shared) == 64
    lookup = {e: {(x["report_id"], x["field_id"]): x for x in data} for e, data in scores.items()}
    q1 = []
    paired = []
    for e, k in [("E1", 1), ("E2", 4), ("E3", 8), ("E4", 16), ("E5", 32)]:
        subset = [lookup[e][key] for key in sorted(shared)]
        per_video = [
            {
                "report_id": rid,
                "correct": sum(x["correct"] for x in subset if x["report_id"] == rid),
                "n": 32,
            }
            for rid in PILOT_IDS
        ]
        correct = sum(x["correct"] for x in subset)
        selected_calls = [
            c
            for c in all_calls
            if c["experiment"] == e and any((c["report_id"], f) in shared for f in c["field_ids"])
        ]
        cost = sum(c["estimated_cost_usd"] for c in selected_calls)
        input_tokens = sum(c["usage"]["input_tokens"] for c in selected_calls)
        output_tokens = sum(c["usage"]["output_tokens"] for c in selected_calls)
        baseline_correct = sum(lookup["E1"][key]["correct"] for key in shared)
        q1.append(
            {
                "k": k,
                "experiment": e,
                "n": 64,
                "correct": correct,
                "video_macro_accuracy": statistics.mean(x["correct"] / 32 for x in per_video),
                "paired_change_vs_k1_pp": (correct - baseline_correct) / 64 * 100,
                "calls": len(selected_calls),
                "cost_total_usd": cost,
                "mean_cost_per_32q_report_usd": cost / 2,
                "mean_input_tokens_per_32q_report": input_tokens / 2,
                "mean_output_tokens_per_32q_report": output_tokens / 2,
                "api_round_trip_sum_s": sum(c["api_round_trip_s"] for c in selected_calls),
                "per_video": per_video,
            }
        )
        for key in sorted(shared):
            paired.append(
                {
                    "report_id": key[0],
                    "field_id": key[1],
                    "k": k,
                    "single": lookup["E1"][key],
                    "batched": lookup[e][key],
                }
            )
    q2 = []
    for subtype in ["loc", "how", "why", "weight"]:
        subset = [x for x in scores["E1"] if x["field_id"].startswith(subtype + "_")]
        byvideo = []
        for rid in PILOT_IDS:
            current = [x for x in subset if x["report_id"] == rid]
            if current:
                byvideo.append(
                    {
                        "report_id": rid,
                        "n": len(current),
                        "correct": sum(x["correct"] for x in current),
                        "accuracy": sum(x["correct"] for x in current) / len(current),
                    }
                )
        q2.append(
            {
                "subtype": subtype,
                "n": len(subset),
                "correct": sum(x["correct"] for x in subset),
                "video_macro_accuracy": statistics.mean(x["accuracy"] for x in byvideo)
                if byvideo
                else None,
                "per_video": byvideo,
            }
        )
    targets = {(rid, fid) for rid in PILOT_IDS for fid in manifest["e6"][rid]["scored_target_ids"]}
    assert len(targets) == 8
    h = lookup["E6_homogeneous"]
    m = lookup["E6_mixed"]
    q3 = {
        "n": 8,
        "homogeneous_correct": sum(h[k]["correct"] for k in targets),
        "mixed_correct": sum(m[k]["correct"] for k in targets),
        "correct_to_wrong": sum(h[k]["correct"] and not m[k]["correct"] for k in targets),
        "wrong_to_correct": sum(not h[k]["correct"] and m[k]["correct"] for k in targets),
        "per_video": [
            {
                "report_id": rid,
                "homogeneous_correct": sum(h[k]["correct"] for k in targets if k[0] == rid),
                "mixed_correct": sum(m[k]["correct"] for k in targets if k[0] == rid),
                "n": 4,
            }
            for rid in PILOT_IDS
        ],
    }
    q3["paired_change_pp"] = (q3["mixed_correct"] - q3["homogeneous_correct"]) / 8 * 100
    validation = {
        "all_124_calls_successful": True,
        "retries": 0,
        "all_378_answers_valid_complete": all(p["all_calls_complete_json"] for p in summaries),
        "all_sent_prompts_match_new_system_and_snapshots": True,
        "all_wire_video_bytes_match_and_identical_across_conditions": True,
        "all_raw_tokens_and_costs_reconcile": True,
        "all_finish_reasons_stop": True,
        "size32_answers": 64,
        "size32_complete": True,
        "full_baseline_executed": False,
    }
    result = {
        "pilot_ids": PILOT_IDS,
        "questions_in_E1": 90,
        "executed_calls": 124,
        "answer_count": sum(len(x) for x in scores.values()),
        "total_estimated_cost_usd": sum(p["metrics"]["usage"]["cost"] for p in summaries),
        "total_input_tokens": sum(p["metrics"]["usage"]["tokens"]["input"] for p in summaries),
        "total_output_tokens": sum(p["metrics"]["usage"]["tokens"]["output"] for p in summaries),
        "total_execution_wall_s": sum(p["timing"]["execution_wall_s"] for p in summaries),
        "total_api_round_trip_s": sum(c["api_round_trip_s"] for c in all_calls),
        "validation": validation,
        "q1_shared_localization": q1,
        "q2_single_question_subtypes": q2,
        "q3_common_targets": q3,
        "per_experiment": summaries,
    }
    assert result["answer_count"] == 378
    repo = Path(__file__).resolve().parents[1]
    assert all(
        sha(repo / "data/hd_epic_q40_v1" / name) == digest
        for name, digest in manifest["source_hashes"].items()
    )
    dump(out / "pilot_report.json", result)
    dump(out / "cost_validation.json", costchecks)
    for name, data in [("per_call_stats.jsonl", all_calls), ("paired_localization.jsonl", paired)]:
        (out / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data))
    (out / "smoke_test.log").write_text(
        "BASELINE TWO-REPORT PILOT — NEW SYSTEM PROMPT\n"
        + json.dumps(result, indent=2)
        + "\n\n"
        + "\n\n".join(p.read_text() for p in logs)
    )
    text = [
        "# Baseline preparation and two-report pilot",
        "Date: October 17, 2026.",
        "Only reports 014 and 022 were executed. The complete 4,355-call baseline is frozen but has NOT been run.",
        "## Data and frozen inputs",
        "76 reports; 3,212 questions: loc 2,861, how 175, why 160, weight 16. All five-option single choice. GT mappings, IDs, all video paths, probed original-video duration and question timestamps validated. 73 localization32 eligible and 24 subtype-mixing eligible reports.",
        "Seed 20261017; one fixed order. Manifest records source hashes, code/source fingerprints, selected IDs, exclusions, batch membership, target positions and exact rendered request hashes. Original data is unchanged.",
        "## System prompt",
        "```text",
        SYSTEM,
        "```",
        "User prompt is unchanged from smoke test. Only task_name context is permitted. No GT/evidence is included in model-facing construction. API adapter explicitly states that clip starts at 0 seconds.",
        "## Pilot acceptance",
        json.dumps(validation, indent=2),
        f"Total: 124 calls, 378 answers. Estimated API cost US${result['total_estimated_cost_usd']:.8f}. Input {result['total_input_tokens']:,}; output {result['total_output_tokens']:,} tokens. Execution wall time {result['total_execution_wall_s']:.2f} s; API round trips {result['total_api_round_trip_s']:.2f} s.",
        "## Q1 — paired localization only",
        "Each setting scores the same 64 questions (32/report); E1's other 26 questions are excluded from this comparison.",
    ]
    for item in q1:
        text.append(
            f"- k={item['k']}: {item['correct']}/64, {item['video_macro_accuracy']:.1%}; change vs k=1 {item['paired_change_vs_k1_pp']:+.2f} pp; {item['calls']} calls; mean cost/32-question report US${item['mean_cost_per_32q_report_usd']:.8f}; mean input/output tokens {item['mean_input_tokens_per_32q_report']:.1f}/{item['mean_output_tokens_per_32q_report']:.1f}."
        )
    text += [
        "## Q2 — single-question subtype results",
        "Video-macro accuracy gives each report with that subtype equal weight; question totals also shown.",
    ]
    for item in q2:
        acc = (
            f"{item['video_macro_accuracy']:.1%}"
            if item["n"]
            else "not tested in these two reports"
        )
        text.append(
            f"- {item['subtype']}: {item['correct']}/{item['n']}; video-macro accuracy {acc}."
        )
    text += [
        "## Q3 — shared localization targets only",
        json.dumps(q3, indent=2),
        "## Direct-cue review",
        "Full E6 screening found 34 surface flags; all remain inspectable in leakage_flags.json and e6_question_review.md. Targets were selected without GT, using a deterministic surface-cue ranking against H2/W2 companions. This procedure cannot exclude semantic leakage.",
        "Pilot review: 014 flag loc_8247 versus loc_5488 shares 'put down' but refers to kettle versus coffee jug, not a direct target answer. 022 flag loc_8924 versus loc_9299 involves pepper mill actions and may provide contextual/order hints; keep it flagged. Neither pilot mixed condition contains an automatically detected exact target-action/timestamp cue. Shared objects and action-order reasoning remain a possible confound. This is not a claim of zero leakage or independent human visual answerability review.",
        "## Interpretation",
        "This pilot validates the tested execution and accounting pipeline, not a reliable performance trend. Two videos and one ordering are insufficient for conclusions or stable uncertainty estimates. No confidence intervals or full-baseline figures are presented. The pilot contains no weight questions; full E1 will cover all 16.",
        "Price estimates: public standard input $0.30/M, cached input $0.03/M, output including reasoning $2.50/M. Cached counts are input subsets; no double counting. Actual invoice and Azure charges are not verified. Missing cache/reasoning breakdowns are not independent zero measurements. API round-trip time includes upload/processing/download, not pure inference. Execution wall time additionally includes media work and bookkeeping.",
        "## Reproduce / resume",
        "Run from /home/azureuser/Capstone-14. Preparation refuses to overwrite an existing frozen directory. Existing successful pilot calls are preserved on resume; max_attempts=1 and retry_failed=False. Old smoke-test outputs were not reused because the prompt differs.",
        "```bash",
        ".venv/bin/python scripts/prepare_hd_epic_baseline.py --output outputs/hd_epic_baseline_20261017 --video-root /home/azureuser/datasets/hd-epic/Videos --seed 20261017 --probe-videos",
        ".venv/bin/python scripts/run_hd_epic_baseline_pilot.py --freeze outputs/hd_epic_baseline_20261017",
        ".venv/bin/python scripts/run_hd_epic_baseline_pilot.py --freeze outputs/hd_epic_baseline_20261017 --execute",
        ".venv/bin/python scripts/report_hd_epic_baseline_pilot.py --freeze outputs/hd_epic_baseline_20261017",
        "```",
        "## Files",
        "selection_manifest.json; vm_data_audit.json; offline_validation.json; plans/; requests/; rendered_request_samples.jsonl; e6_question_review.md; leakage_flags.json; cost_estimate.json; pilot/pilot_plan.json; report/pilot_report.json; report/per_call_stats.jsonl; report/paired_localization.jsonl; report/smoke_test.log.",
        "All original API JSON responses and base64 request bodies remain under pilot/. The ZIP excludes large wire_inputs and media copies; credentials are excluded.",
    ]
    (out / "pilot_report_en.md").write_text("\n\n".join(text) + "\n")
    zh = [
        "# Baseline 準備與兩份 report pilot",
        "資料核對與 E1–E6 固定清單已完成；只跑 014、022，尚未啟動完整 baseline。",
        f"124 次請求、378 個答案；費用 US${result['total_estimated_cost_usd']:.8f}。Input {result['total_input_tokens']:,}，output {result['total_output_tokens']:,} tokens；執行 {result['total_execution_wall_s']:.2f} 秒。",
        "32 題的兩次請求都完整回傳，64 個答案合法，沒有截斷。所有請求都使用新 system prompt、VM 的固定影片窗口。",
        "## 同一批 64 道定位題的比較",
    ]
    for x in q1:
        zh.append(
            f"- Size {x['k']}：{x['correct']}/64（{x['video_macro_accuracy']:.1%}），相對 size 1 {x['paired_change_vs_k1_pp']:+.2f} 個百分點。"
        )
    zh += [
        "## E6 共同目標",
        f"只計同一批 8 題：純題型 {q3['homogeneous_correct']}/8；混合題型 {q3['mixed_correct']}/8。",
        "這是兩支影片的 pilot，不足以判定題數或題型混合的普遍效果。完整資料保留 34 筆線索標記，不能宣稱沒有語意洩漏。Pilot 沒有 weight 題。",
        "完整輸入輸出：smoke_test.log；逐次成本／tokens／時間：per_call_stats.jsonl；固定清單：上層 selection_manifest.json。",
    ]
    (out / "pilot_report_zh.md").write_text("\n\n".join(zh) + "\n")
    with zipfile.ZipFile(
        root / "baseline_preparation_pilot_evidence.zip", "w", zipfile.ZIP_DEFLATED
    ) as z:
        for p in root.rglob("*"):
            rel = p.relative_to(root)
            if (
                p.is_file()
                and p.suffix != ".zip"
                and "wire_inputs" not in rel.parts
                and "media" not in rel.parts
            ):
                z.write(p, rel)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "executed_calls",
                    "answer_count",
                    "total_estimated_cost_usd",
                    "total_input_tokens",
                    "total_output_tokens",
                    "validation",
                    "q1_shared_localization",
                    "q3_common_targets",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
