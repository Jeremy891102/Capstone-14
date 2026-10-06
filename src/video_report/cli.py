"""Command-line interface: ``video-report {run,resume,status,evaluate,validate-data}``.

Exit codes: 0 completed / success, 2 invalid input or incompatible resume, 3 run finished with
failed calls, 4 run interrupted (resume to continue), 5 run locked by another writer.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from types import FrameType
from typing import Any

from video_report.config import ConfigError, PricingConfig
from video_report.datasets.base import (
    DataValidationError,
    choose_video_root,
    load_benchmark,
    load_ground_truth,
    load_reports,
    missing_local_videos,
    resolve_video,
    validate_targets,
)
from video_report.io_utils import read_json
from video_report.prompting import PromptError
from video_report.providers.base import ProviderError
from video_report.run_store import RunDir, RunError, RunLockedError

EXIT_OK, EXIT_INPUT, EXIT_FAILURES, EXIT_INTERRUPTED, EXIT_LOCKED = 0, 2, 3, 4, 5
_STATE_EXIT = {
    "completed": EXIT_OK,
    "completed_with_failures": EXIT_FAILURES,
    "interrupted": EXIT_INTERRUPTED,
}


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def _sigterm_as_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def _progress(state: dict[str, Any]) -> None:
    print(
        f"[{state['status']}] {state['call_id']} (attempts: {len(state['attempts'])})", flush=True
    )


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "concurrency": args.concurrency,
        "max_attempts": args.max_attempts,
        "timeout_s": args.timeout_s,
    }


def _add_exec_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--concurrency", type=int, help="override execution.concurrency")
    p.add_argument("--max-attempts", type=int, help="override execution.max_attempts")
    p.add_argument("--timeout-s", type=float, help="override execution.timeout_s")
    p.add_argument("--quiet", action="store_true", help="no per-call progress lines")


def cmd_run(args: argparse.Namespace) -> int:
    from video_report.pipeline import create_run, execute_run

    run = create_run(
        Path(args.config),
        Path(args.runs_dir),
        run_id=args.run_id,
        video_root=args.video_root,
        overrides=_overrides(args),
    )
    print(f"created run: {run.path}", flush=True)
    result = execute_run(
        run,
        mode="create",
        overrides=_overrides(args),
        on_call_done=None if args.quiet else _progress,
    )
    _print({"run": str(run.path), "state": result.state, "counts": result.counts})
    return _STATE_EXIT[result.state]


def cmd_resume(args: argparse.Namespace) -> int:
    from video_report.pipeline import check_resume_compatibility, execute_run

    run = RunDir(Path(args.run))
    if not run.run_json.is_file():
        raise RunError(f"not a run directory: {run.path}")
    notes = check_resume_compatibility(
        run,
        config_path=Path(args.config) if args.config else None,
        video_root=args.video_root,
        allow_code_change=args.allow_code_change,
    )
    for n in notes:
        print(f"note: {n}")
    result = execute_run(
        run,
        mode="resume",
        overrides=_overrides(args),
        retry_failed=args.retry_failed,
        on_call_done=None if args.quiet else _progress,
        session_notes=notes,
    )
    _print(
        {
            "run": str(run.path),
            "state": result.state,
            "counts": result.counts,
            "recovered": result.recovered,
            "executed_calls": result.executed_calls,
        }
    )
    return _STATE_EXIT[result.state]


def cmd_status(args: argparse.Namespace) -> int:
    from video_report.pipeline import run_status

    _print(run_status(RunDir(Path(args.run))))
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace) -> int:
    from video_report.evaluation.evaluate import evaluate_run

    pricing = PricingConfig.model_validate(read_json(Path(args.pricing))) if args.pricing else None
    out_dir, metrics = evaluate_run(
        Path(args.run),
        Path(args.ground_truth),
        allow_markdown_fence=args.allow_markdown_fence,
        pricing=pricing,
    )
    summary = {
        "evaluation_dir": str(out_dir),
        "provisional": metrics["provisional"],
        "provisional_reasons": metrics["provisional_reasons"],
        "warnings": metrics["warnings"],
        "field_accuracy": metrics["fields"]["field_accuracy"],
        "field_outcomes": metrics["fields"]["outcomes"],
        "report_completeness_rate": metrics["reports"]["report_completeness_rate"],
        "all_fields_correct_rate": metrics["reports"]["all_fields_correct_rate"],
        "execution_completion": metrics["execution"]["execution_completion"],
        "cost": metrics["usage"]["cost"],
        "cost_status": metrics["usage"]["cost_status"],
    }
    _print(summary)
    return EXIT_OK


def cmd_validate_data(args: argparse.Namespace) -> int:
    bench = load_benchmark(Path(args.benchmark))
    reports = load_reports(bench.reports_path)
    out: dict[str, Any] = {
        "benchmark_id": bench.manifest.benchmark_id,
        "reports": len(reports),
        "fields": sum(len(r.fields) for r in reports),
    }
    gt_path = Path(args.ground_truth) if args.ground_truth else bench.ground_truth_path
    if gt_path is not None and gt_path.is_file():
        selection = {r.id: [f.id for f in r.fields] for r in reports}
        validate_targets(reports, load_ground_truth(gt_path), selection)
        out["ground_truth"] = f"valid for all {out['fields']} fields"
    else:
        out["ground_truth"] = "not checked (no file)"
    if args.check_videos:
        root, source = choose_video_root(args.video_root, None, bench)
        vids = [
            resolve_video(v.id, v.uri, v.mime_type, root) for r in reports for v in r.input.videos
        ]
        missing = missing_local_videos(vids)
        out["video_root"] = {"path": str(root) if root else None, "source": source}
        out["videos"] = {"checked": len(vids), "missing": missing}
        _print(out)
        return EXIT_INPUT if missing else EXIT_OK
    _print(out)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-report")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="create a new run from an experiment config and execute it")
    p.add_argument("--config", required=True)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--run-id")
    p.add_argument("--video-root")
    _add_exec_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resume", help="continue an existing run from its frozen snapshot")
    p.add_argument("run")
    p.add_argument("--config", help="verify the run still matches these source inputs")
    p.add_argument("--video-root", help="only used with --config for the compatibility check")
    p.add_argument("--retry-failed", action="store_true", help="also re-run failed calls")
    p.add_argument("--allow-code-change", action="store_true")
    _add_exec_args(p)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("status", help="show run state and call counts")
    p.add_argument("run")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("evaluate", help="score a run offline (no model calls)")
    p.add_argument("--run", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--allow-markdown-fence", action="store_true")
    p.add_argument("--pricing", help="JSON file overriding the run's pricing config")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("validate-data", help="validate a prepared benchmark directory")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--ground-truth")
    p.add_argument("--check-videos", action="store_true")
    p.add_argument("--video-root")
    p.set_defaults(func=cmd_validate_data)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _sigterm_as_interrupt)
    try:
        rc: int = args.func(args)
        return rc
    except RunLockedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_LOCKED
    except (
        ConfigError,
        DataValidationError,
        PromptError,
        RunError,
        ProviderError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT


if __name__ == "__main__":
    sys.exit(main())
