"""Readable run evidence, generated offline from frozen inputs and checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path

from video_report.evaluation.aggregate import _usage_block
from video_report.io_utils import atomic_write_text, read_json
from video_report.providers.base import redact
from video_report.run_store import RunDir


def export_run_log(run: RunDir, evaluation_dir: Path | None = None) -> Path:
    """Write a derived log; never call a provider or modify inference evidence."""
    cfg = run.load_config()
    calls = {c["call_id"]: c for c in run.read_all_calls()}
    lines = [
        "VIDEO REPORT SMOKE TEST LOG",
        "Derived from saved evidence. No API requests are made to export this log.",
        "",
        "RUN",
        json.dumps(run.read_meta(), indent=2, ensure_ascii=False),
        "",
        "PRICING",
        json.dumps(cfg.pricing.model_dump() if cfg.pricing else None, indent=2),
        "Cost is a public-price estimate, not an invoice. Unknown usage is not zero.",
        "",
        "TIME DEFINITIONS",
        "latency_s includes provider media preparation and API round trip.",
        "It is not backend-only model inference time, which is not supplied.",
        "",
    ]
    for request in run.load_requests().values():
        state = calls[request.call_id]
        lines.extend(
            [
                f"CALL {request.call_id}",
                "INPUT (exact frozen provider request; media referenced by path/range)",
                json.dumps(request.to_json(), indent=2, ensure_ascii=False),
                "CHECKPOINT / ATTEMPTS / TIME / USAGE",
                json.dumps(state, indent=2, ensure_ascii=False),
                "CALL USAGE / ESTIMATED COST",
                json.dumps(_usage_block(state["attempts"], cfg.pricing), indent=2),
            ]
        )
        for attempt in state["attempts"]:
            response_file = attempt.get("response_file")
            if response_file:
                lines.extend(
                    [
                        f"OUTPUT: attempt {attempt['attempt']} ({response_file})",
                        json.dumps(run.read_response(response_file), indent=2, ensure_ascii=False),
                    ]
                )
        lines.append("")
    attempts = [a for state in calls.values() for a in state["attempts"]]
    lines.extend(
        ["TOTAL USAGE / ESTIMATED COST", json.dumps(_usage_block(attempts, cfg.pricing), indent=2)]
    )
    if evaluation_dir is not None:
        lines.extend(["", "OFFLINE EVALUATION", str(evaluation_dir)])
        for name in ("evaluation.json", "call_parses.jsonl", "field_scores.jsonl", "metrics.json"):
            lines.extend([name, (evaluation_dir / name).read_text(encoding="utf-8")])
        effective = read_json(evaluation_dir / "evaluation.json")["scoring"]
        lines.extend(
            [
                "Evaluation may override pricing; metrics.json carries its effective cost.",
                json.dumps(effective, indent=2),
            ]
        )
    else:
        lines.extend(["", "Evaluation not yet run; ground-truth scoring is pending."])
    secrets = tuple(v for k, v in os.environ.items() if k.endswith("API_KEY") and v)
    text = redact("\n".join(lines) + "\n", extra_secrets=secrets)
    target = (evaluation_dir if evaluation_dir is not None else run.path) / "smoke_test.log"
    atomic_write_text(target, text)
    return target
