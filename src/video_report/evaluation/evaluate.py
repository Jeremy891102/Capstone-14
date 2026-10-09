"""Offline evaluation of a run: parse saved responses, score against ground truth, aggregate.

Makes no provider calls and never modifies inference evidence. Each evaluation is written to a
new directory ``runs/<run_id>/evaluations/<eval_id>/`` with its scoring configuration, parser
and scorer versions, and the ground-truth fingerprint.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from video_report.config import PricingConfig
from video_report.datasets.base import load_ground_truth, validate_targets
from video_report.evaluation.aggregate import aggregate
from video_report.evaluation.parser import PARSER_VERSION, parse_response
from video_report.evaluation.scorers import SCORER_VERSION, FieldScore, score_field
from video_report.io_utils import (
    atomic_write_json,
    fingerprint,
    sha256_file,
    utc_now_iso,
    write_jsonl,
)
from video_report.run_store import RunDir


def evaluate_run(
    run_path: Path,
    ground_truth_path: Path,
    *,
    allow_markdown_fence: bool = False,
    pricing: PricingConfig | None = None,
    eval_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    run = RunDir(run_path)
    fingerprints = run.verify_snapshot()
    cfg = run.load_config()
    reports = run.load_reports()
    by_report = {r.id: r for r in reports}
    selection = {r.id: [f.id for f in r.fields] for r in reports}
    # Frozen reports hold only the selected fields, so other fields' targets are skipped.
    targets = validate_targets(
        reports, load_ground_truth(ground_truth_path), selection, reject_unknown_fields=False
    )
    effective_pricing = pricing if pricing is not None else cfg.pricing
    writer_active = run.lock_is_held()

    calls = run.read_all_calls()
    scores: list[FieldScore] = []
    call_rows = []
    for state in calls:
        report = by_report[state["report_id"]]
        parsed = None
        reason = None
        if state["status"] == "succeeded":
            final = state["attempts"][state["final_attempt"] - 1]
            text = run.read_response(final["response_file"])["response"]["text"]
            parsed = parse_response(
                text, state["field_ids"], allow_markdown_fence=allow_markdown_fence
            )
            call_rows.append(
                {
                    "call_id": state["call_id"],
                    "status": "succeeded",
                    "parse_status": parsed.status,
                    "missing": list(parsed.missing),
                    "extra_keys": list(parsed.extra_keys),
                    "detail": parsed.detail,
                }
            )
        else:
            reason = f"call {state['status']}" + (
                f" ({state['failure']})" if state.get("failure") else ""
            )
            call_rows.append({"call_id": state["call_id"], "status": state["status"]})
        for fid in state["field_ids"]:
            scores.append(
                score_field(
                    report.field(fid), targets[(report.id, fid)], state["call_id"], parsed, reason
                )
            )

    metrics = aggregate(scores, calls, effective_pricing, writer_active=writer_active)

    eid = eval_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
    )
    out_dir = run.evaluations_dir / eid
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir()  # exclusive; never overwrite a previous evaluation
    meta = run.read_meta()
    evaluation = {
        "eval_id": eid,
        "created_at": utc_now_iso(),
        "run_id": meta["run_id"],
        "run_state_at_evaluation": meta["state"],
        "run_fingerprint": fingerprints["overall"],
        "scoring": {
            "parser_version": PARSER_VERSION,
            "scorer_version": SCORER_VERSION,
            "allow_markdown_fence": allow_markdown_fence,
            "pricing": effective_pricing.model_dump() if effective_pricing else None,
            "pricing_from": "override" if pricing is not None else "frozen run config",
        },
        "ground_truth": {
            "path": str(ground_truth_path.resolve()),
            "file_sha256": sha256_file(ground_truth_path),
            "selected_targets_fingerprint": fingerprint(
                sorted([t.report_id, t.field_id, t.target, t.target_text] for t in targets.values())
            ),
            "num_selected_targets": len(targets),
        },
    }
    atomic_write_json(out_dir / "evaluation.json", evaluation)
    write_jsonl(out_dir / "field_scores.jsonl", [s.to_json() for s in scores])
    write_jsonl(out_dir / "call_parses.jsonl", call_rows)
    atomic_write_json(out_dir / "metrics.json", metrics)
    from video_report.run_log import export_run_log

    export_run_log(run, out_dir)
    return out_dir, metrics
