#!/usr/bin/env python3
"""No-video control for the 2026-10-20 HD-EPIC baseline.

Re-sends the baseline's frozen requests for E1 (1 question per call), E3 (8) and E5 (32)
with the video removed and every other field unchanged: system prompt, user prompt, batches,
question order, model and generation settings. Then scores the answers and pairs them with
the baseline's with-video results, question by question.

The baseline folder is only read, never written. Results go to outputs/no_video_control/
(git-ignored). Default is a dry run; paid calls require --execute.

    set -a; source .env; set +a
    PYTHONPATH=src python scripts/run_no_video_control.py                     # dry run
    PYTHONPATH=src python scripts/run_no_video_control.py --limit 5 --execute # 5 calls per experiment
    PYTHONPATH=src python scripts/run_no_video_control.py --execute --max-cost-usd 2
    PYTHONPATH=src python scripts/run_no_video_control.py --score
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from video_report.evaluation.parser import parse_response
from video_report.providers.base import (
    ModelRequest,
    PermanentProviderError,
    RetryableProviderError,
)

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "results/hd_epic_q40_v1_gemini35_flash_lite_20261020"
REQUESTS = BASELINE / "data/frozen_requests"
GROUND_TRUTH = BASELINE / "data/frozen_ground_truth"
PER_FIELD = BASELINE / "data/per_field.csv"
DEFAULT_OUT = REPO / "outputs/no_video_control"

EXPERIMENTS = {"E1": 1, "E3": 8, "E5": 32}  # questions per call
# Gemini 3.5 Flash-Lite, USD per token (Gemini API = Vertex global endpoint; checked 2026-10-21).
PRICE_IN, PRICE_OUT = 0.30 / 1e6, 2.50 / 1e6
LETTERS = {"A", "B", "C", "D", "E"}
CATEGORY = {
    "fine_grained_action_localization": "localization",
    "fine_grained_how_recognition": "how",
    "fine_grained_why_recognition": "why",
    "ingredient_ingredient_weight": "weight",
}


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_requests(experiment: str) -> list[dict[str, Any]]:
    return read_jsonl_gz(REQUESTS / f"{experiment}.jsonl.gz")


def without_video(frozen: dict[str, Any]) -> ModelRequest:
    """The frozen request exactly as sent on 2026-10-20, minus its video."""
    return ModelRequest(
        call_id=frozen["call_id"],
        report_id=frozen["report_id"],
        field_ids=tuple(frozen["field_ids"]),
        system=frozen["system"],
        user=frozen["user"],
        videos=(),
        model=frozen["model"],
        generation=frozen["generation"],
        fps=frozen["fps"],
    )


def check_out_dir(out: Path) -> Path:
    out = out.resolve()
    if out == BASELINE.resolve() or BASELINE.resolve() in out.parents:
        raise SystemExit(f"refusing to write inside the baseline folder: {out}")
    if (REPO / "results").resolve() in out.parents:
        raise SystemExit(f"refusing to write inside results/: {out}")
    return out


def done_calls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {r["call_id"] for r in map(json.loads, f) if r["status"] == "ok"}


def estimate_usd(requests: list[ModelRequest]) -> float:
    # Text-only input ≈ 114 + 254.375 tokens per question (the baseline runner's formula without
    # the video term); output ≈ 12 tokens per question. Rough, for the dry run only.
    n = sum(len(r.field_ids) for r in requests)
    return (114 * len(requests) + 254.375 * n) * PRICE_IN + 12 * n * PRICE_OUT


def cost_usd(usage: dict[str, Any] | None) -> float:
    if not usage:
        return 0.0
    out = (usage.get("output_tokens") or 0) + (usage.get("reasoning_tokens") or 0)
    return (usage.get("input_tokens") or 0) * PRICE_IN + out * PRICE_OUT


def call_once(provider: Any, request: ModelRequest, attempts: int = 3) -> dict[str, Any]:
    record: dict[str, Any] = {"call_id": request.call_id, "field_ids": list(request.field_ids)}
    for attempt in range(1, attempts + 1):
        try:
            response = provider.generate(request, timeout_s=120)
        except RetryableProviderError as exc:
            if attempt < attempts:
                time.sleep(2**attempt)
                continue
            return record | {"status": "error", "error": repr(exc), "attempts": attempt}
        except PermanentProviderError as exc:
            return record | {"status": "error", "error": repr(exc), "attempts": attempt}
        return record | {"status": "ok", "attempts": attempt} | response.to_json()
    raise AssertionError("unreachable")


def execute(
    experiment: str,
    requests: list[ModelRequest],
    provider: Any,
    out: Path,
    *,
    budget: dict[str, float],
    concurrency: int,
) -> None:
    path = out / f"{experiment}.jsonl"
    finished = done_calls(path)
    todo = [r for r in requests if r.call_id not in finished]
    print(f"{experiment}: {len(finished)} done, {len(todo)} to run")
    lock = threading.Lock()

    def work(request: ModelRequest) -> None:
        with lock:
            if budget["spent"] >= budget["cap"]:
                return
        record = call_once(provider, request)
        record |= {"experiment": experiment, "time": datetime.now(timezone.utc).isoformat()}
        with lock:
            budget["spent"] += cost_usd(record.get("usage"))
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(work, todo))
    if budget["spent"] >= budget["cap"]:
        print(f"stopped at the US${budget['cap']:.2f} cap; rerun to continue")


# --------------------------------------------------------------------------- scoring


def score(out: Path) -> dict[str, Any]:
    with PER_FIELD.open(encoding="utf-8") as f:
        baseline = {(r["experiment"], r["report_id"], r["field_id"]): r for r in csv.DictReader(f)}
    rows = []
    missing = {}
    for experiment in EXPERIMENTS:
        targets = {
            (t["report_id"], t["field_id"]): t["target"]
            for t in read_jsonl_gz(GROUND_TRUTH / f"{experiment}.jsonl.gz")
        }
        path = out / f"{experiment}.jsonl"
        records = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for r in map(json.loads, f):
                    if r["status"] == "ok":
                        records[r["call_id"]] = r
        planned = 0
        for frozen in load_requests(experiment):
            planned += len(frozen["field_ids"])
            record = records.get(frozen["call_id"])
            if record is None:
                continue
            parsed = parse_response(record.get("text"), frozen["field_ids"])
            for field_id in frozen["field_ids"]:
                answer = parsed.answers.get(field_id) if parsed.status == "ok" else None
                valid = isinstance(answer, str) and answer.strip().upper() in LETTERS
                predicted = answer.strip().upper() if valid else None
                key = (frozen["report_id"], field_id)
                base = baseline[(experiment, *key)]
                rows.append(
                    {
                        "experiment": experiment,
                        "report_id": key[0],
                        "field_id": field_id,
                        "category": CATEGORY[base["category"]],
                        "matched_localization": base["matched_localization"] == "True",
                        "source_video": base["source_video"],
                        "target": targets[key],
                        "predicted_no_video": predicted,
                        "valid_no_video": valid,
                        "correct_no_video": predicted == targets[key],
                        "correct_with_video": base["correct"] == "True",
                    }
                )
        missing[experiment] = planned - sum(1 for r in rows if r["experiment"] == experiment)
    return {"rows": rows, "missing": missing}


def video_weighted(rows: list[dict[str, Any]], column: str) -> float:
    by_video: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_video[r["source_video"]].append(r[column])
    return 100 * sum(sum(v) / len(v) for v in by_video.values()) / len(by_video)


def bootstrap_gap(rows: list[dict[str, Any]], reps: int = 2000, seed: int = 20261021):
    """With-video minus no-video, video-weighted; 95% interval from resampling videos."""
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_video[r["source_video"]].append(r)
    per_video = [
        sum(r["correct_with_video"] for r in rs) / len(rs)
        - sum(r["correct_no_video"] for r in rs) / len(rs)
        for rs in by_video.values()
    ]
    rng = random.Random(seed)
    gaps = sorted(
        100 * sum(rng.choices(per_video, k=len(per_video))) / len(per_video) for _ in range(reps)
    )
    return gaps[int(0.025 * reps)], gaps[int(0.975 * reps) - 1]


def summarize(result: dict[str, Any], out: Path) -> None:
    rows = result["rows"]
    if not rows:
        raise SystemExit("nothing to score yet; run with --execute first")
    with (out / "per_field_no_video.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = []
    print("\nLocalization, same targets in every condition (video-weighted %)")
    print(f"{'per call':>8}  {'with video':>10}  {'no video':>8}  {'gap':>6}  95% interval  invalid")
    for experiment, size in EXPERIMENTS.items():
        sub = [r for r in rows if r["experiment"] == experiment and r["matched_localization"]]
        if not sub:
            continue
        with_v, no_v = video_weighted(sub, "correct_with_video"), video_weighted(sub, "correct_no_video")
        lo, hi = bootstrap_gap(sub)
        invalid = 100 * sum(not r["valid_no_video"] for r in sub) / len(sub)
        print(f"{size:>8}  {with_v:>10.1f}  {no_v:>8.1f}  {with_v - no_v:>+6.1f}  [{lo:+.1f}, {hi:+.1f}]  {invalid:.1f}%")
        lines.append({"experiment": experiment, "per_call": size, "targets": len(sub),
                      "with_video": with_v, "no_video": no_v, "gap": with_v - no_v,
                      "gap_ci95": [lo, hi], "invalid_pct": invalid})
    print("\nOne question per call, by question type (video-weighted %)")
    by_type = []
    for category in ["localization", "how", "why", "weight"]:
        sub = [r for r in rows if r["experiment"] == "E1" and r["category"] == category]
        if not sub:
            continue
        with_v, no_v = video_weighted(sub, "correct_with_video"), video_weighted(sub, "correct_no_video")
        lo, hi = bootstrap_gap(sub)
        videos = len({r["source_video"] for r in sub})
        print(f"{category:>12}  with video {with_v:5.1f}  no video {no_v:5.1f}  gap {with_v - no_v:+5.1f} "
              f"[{lo:+.1f}, {hi:+.1f}]  ({len(sub)} questions, {videos} videos)")
        by_type.append({"category": category, "questions": len(sub), "videos": videos,
                        "with_video": with_v, "no_video": no_v, "gap": with_v - no_v,
                        "gap_ci95": [lo, hi]})
    if any(result["missing"].values()):
        print(f"\nnot yet answered (excluded): {result['missing']}")
    summary = {"by_batch_size": lines, "by_type_e1": by_type, "missing": result["missing"],
               "chance_pct": 20.0, "baseline": str(BASELINE.relative_to(REPO))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'per_field_no_video.csv'} and {out / 'summary.json'}")


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None, provider: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--execute", action="store_true", help="make paid API calls")
    parser.add_argument("--limit", type=int, default=None, help="first N calls per experiment")
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--score", action="store_true", help="score what has been run so far")
    args = parser.parse_args(argv)
    out = check_out_dir(args.out)

    if args.score:
        summarize(score(out), out)
        return 0

    plan = {e: [without_video(r) for r in load_requests(e)][: args.limit] for e in args.experiments}
    total = sum(estimate_usd(rs) for rs in plan.values())
    for e, rs in plan.items():
        print(f"{e}: {len(rs)} calls, {sum(len(r.field_ids) for r in rs)} questions, "
              f"{EXPERIMENTS[e]} per call, ≈ US${estimate_usd(rs):.2f}")
    print(f"total ≈ US${total:.2f} (rough); cap US${args.max_cost_usd:.2f}")
    if not args.execute:
        example = next(iter(plan.values()))[0]
        print("\nexample request (videos removed):")
        print(json.dumps(example.to_json(), ensure_ascii=False, indent=2)[:1500])
        print("\ndry run only; add --execute to make paid calls")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    if provider is None:
        from video_report.config import GeminiProviderConfig
        from video_report.providers.vertex_gemini import VertexGeminiProvider

        models = {r.model for rs in plan.values() for r in rs}
        assert len(models) == 1, models
        try:
            provider = VertexGeminiProvider(models.pop(), GeminiProviderConfig(backend="vertex_express"))
        except PermanentProviderError as exc:
            raise SystemExit(f"{exc}; run: set -a; source .env; set +a") from None
    budget = {"spent": 0.0, "cap": args.max_cost_usd}
    for e, rs in plan.items():
        execute(e, rs, provider, out, budget=budget, concurrency=args.concurrency)
    print(f"spent ≈ US${budget['spent']:.2f} this session; results in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
