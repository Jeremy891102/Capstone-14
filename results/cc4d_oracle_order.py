"""Re-score CC4D Qwen oracle clips by whether their only error is order.

Usage: python results/cc4d_oracle_order.py RUNS_DIR RUN [RUN ...] [--out DIR]

RUN arguments are oracle, k=8, and k=all in that order. Each resolves to
RUNS_DIR/RUN/evaluations/<newest>/field_scores.jsonl. No model calls are made.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

SEED = 20261124
BOOTSTRAPS = 2000
ERROR_TYPES = ("Technique", "Preparation", "Measurement", "Timing", "Temperature", "Order")
EXPECTED_GROUPS = {
    "done correctly": 411,
    "Order only": 104,
    "Order + other": 94,
    "no Order": 342,
}
EXPECTED = {
    "oracle": (57.7, 14.6, 18.5, 87.6, 53.0),
    "k8": (51.6, 63.6, 67.6, 39.4, 53.5),
    "kall": (46.3, 88.2, 90.0, 13.0, 51.5),
}
FIELD_RE = re.compile(r"step(\d+)_(correct|error_types)\Z")


def step_group(target_correct: bool, errors: list[str]) -> str:
    """Map a performed step's GT to the four mutually exclusive analysis groups."""
    tags = {str(e).removesuffix(" Error") for e in errors}
    if target_correct:
        return "done correctly"
    if tags == {"Order"}:
        return "Order only"
    if "Order" in tags:
        return "Order + other"
    return "no Order"


def visible_target(target_correct: bool, group: str) -> bool:
    """Treat Order-only errors as visually correct; all other labels stay unchanged."""
    return bool(target_correct or group == "Order only")


def strip_order_missing(values: Any) -> list[str]:
    """Remove options F (Order) and G (Missing Step), retaining normalized labels."""
    if values is None:
        return []
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except json.JSONDecodeError:
            values = [values]
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    return sorted({str(v).strip() for v in values if str(v).strip() not in {"F", "G"}})


def as_labels(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except json.JSONDecodeError:
            values = [values]
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    return sorted({str(v).strip() for v in values if str(v).strip()})


def exact_match_without_fg(predicted: Any, target: Any) -> bool:
    return strip_order_missing(predicted) == strip_order_missing(target)


def parsed_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def bool_is_correct(predicted: Any, target: bool) -> bool:
    return parsed_bool(predicted) is target


def recording_id(report_id: str) -> str:
    return report_id.split("__", 1)[0]


def read_ground_truth(
    path: Path, eligible_steps: set[tuple[str, int]] | None = None
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, int]]:
    steps: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.open():
        row = json.loads(line)
        match = FIELD_RE.fullmatch(row.get("field_id", ""))
        if not match or match.group(2) != "correct":
            continue
        step_key = (row["report_id"], int(match.group(1)))
        if eligible_steps is not None and step_key not in eligible_steps:
            continue
        evidence = row.get("evidence") or {}
        errors = [e.get("tag", "") for e in evidence.get("errors", [])]
        group = step_group(bool(row["target"]), errors)
        steps[step_key] = {
            "target": bool(row["target"]),
            "errors": errors,
            "group": group,
        }
    counts = {g: sum(s["group"] == g for s in steps.values()) for g in EXPECTED_GROUPS}
    return steps, counts


def newest_scores(runs_dir: Path, run: str) -> Path:
    files = sorted(glob.glob(str(runs_dir / run / "evaluations" / "*" / "field_scores.jsonl")))
    if not files:
        raise FileNotFoundError(f"no evaluations/*/field_scores.jsonl for run {run!r}")
    return Path(files[-1])


def read_run(path: Path, oracle: bool) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows = {}
    for line in path.open():
        row = json.loads(line)
        match = FIELD_RE.fullmatch(row.get("field_id", ""))
        if not match or row.get("extra", {}).get("not_done_item") is not False:
            continue
        report = recording_id(row["report_id"]) if oracle else row["report_id"]
        key = (report, int(match.group(1)), match.group(2))
        rows[key] = {
            "predicted": row.get("predicted"),
            "target": row.get("target"),
            "correct": bool(row.get("correct")),
        }
    return rows


def percentile_ci(
    metric: Callable[[list[tuple[str, int, str]]], float],
    keys: list[tuple[str, int, str]],
    rng: np.random.Generator,
    draws: int = BOOTSTRAPS,
) -> tuple[float, float, float]:
    recordings: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for key in keys:
        recordings[key[0]].append(key)
    ids = sorted(recordings)
    estimate = metric(keys)
    values = []
    for _ in range(draws):
        chosen = rng.choice(ids, size=len(ids), replace=True)
        sample = [key for rec in chosen for key in recordings[rec]]
        values.append(metric(sample))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(estimate), float(low), float(high)


def _rate(rows: dict, keys: list, fn: Callable[[tuple, dict], bool]) -> float:
    return float(np.mean([fn(k, rows[k]) for k in keys]))


def analyze(
    runs: dict[str, dict], gt: dict[tuple[str, int], dict], keys: list, rng: np.random.Generator
) -> list[dict[str, Any]]:
    output = []

    def add(run_name: str, metric: str, selected: list, fn: Callable[[tuple, dict], bool], kind="correct"):
        data = runs[run_name]
        sub = [k for k in selected if k in data]
        if not sub:
            return
        vals = percentile_ci(lambda ks: 100 * _rate(data, ks, fn), sub, rng)
        output.append(dict(run=run_name, task=kind, metric=metric, n=len(sub), recordings=len({k[0] for k in sub}), value=vals[0], ci_low=vals[1], ci_high=vals[2]))

    correct_keys = [k for k in keys if k[2] == "correct" and (k[0], k[1]) in gt]
    for run_name, data in runs.items():
        sub = [k for k in correct_keys if k in data]
        for label, target_fn in (("standard", lambda k: gt[k[0], k[1]]["target"]), ("visible label", lambda k: visible_target(gt[k[0], k[1]]["target"], gt[k[0], k[1]]["group"]))):
            add(run_name, f"accuracy ({label})", sub, lambda k, r, t=target_fn: bool_is_correct(r["predicted"], t(k)))
            add(run_name, f"says correct ({label})", sub, lambda k, r: parsed_bool(r["predicted"]) is True)
            for group, condition in (("steps labeled correct", lambda g: g == "done correctly"), ("steps labeled wrong", lambda g: g != "done correctly")):
                if label == "visible label":
                    condition = lambda g: g not in ("Order only", "done correctly") if group == "steps labeled wrong" else g in ("Order only", "done correctly")
                ks = [k for k in sub if condition(gt[k[0], k[1]]["group"])]
                add(run_name, f"accuracy on {group} ({label})", ks, lambda k, r, t=target_fn: bool_is_correct(r["predicted"], t(k)))
            # Balanced accuracy is the unweighted mean of accuracy within the two classes.
            def balanced(sample, data=data, target_fn=target_fn, label=label):
                positives = [k for k in sample if target_fn(k)]
                if label == "visible label":
                    positives = [k for k in sample if visible_target(gt[k[0], k[1]]["target"], gt[k[0], k[1]]["group"])]
                negatives = [k for k in sample if k not in positives]
                if not positives or not negatives:
                    return float("nan")
                return 50 * (_rate(data, positives, lambda k, r: bool_is_correct(r["predicted"], target_fn(k))) + _rate(data, negatives, lambda k, r: bool_is_correct(r["predicted"], target_fn(k))))
            val = percentile_ci(balanced, sub, rng)
            output.append(dict(run=run_name, task="correct", metric=f"balanced accuracy ({label})", n=len(sub), recordings=len({k[0] for k in sub}), value=val[0], ci_low=val[1], ci_high=val[2]))
        for group in EXPECTED_GROUPS:
            ks = [k for k in sub if gt[k[0], k[1]]["group"] == group]
            add(run_name, f"accuracy: {group}", ks, lambda k, r: bool_is_correct(r["predicted"], gt[k[0], k[1]]["target"]))
        for error in ERROR_TYPES:
            ks = [k for k in sub if not gt[k[0], k[1]]["target"] and any(error in str(tag) for tag in gt[k[0], k[1]]["errors"])]
            add(run_name, f"accuracy: wrong with {error}", ks, lambda k, r: bool_is_correct(r["predicted"], gt[k[0], k[1]]["target"]))

        def balanced_for(sample, visible):
            positive = [
                k for k in sample
                if (visible_target(gt[k[0], k[1]]["target"], gt[k[0], k[1]]["group"])
                    if visible else gt[k[0], k[1]]["target"])
            ]
            negative = [k for k in sample if k not in positive]
            if not positive or not negative:
                return float("nan")
            target = lambda k: visible_target(gt[k[0], k[1]]["target"], gt[k[0], k[1]]["group"]) if visible else gt[k[0], k[1]]["target"]
            return 50 * (
                _rate(data, positive, lambda k, r: bool_is_correct(r["predicted"], target(k)))
                + _rate(data, negative, lambda k, r: bool_is_correct(r["predicted"], target(k)))
            )

        change = percentile_ci(
            lambda sample: balanced_for(sample, True) - balanced_for(sample, False),
            sub,
            rng,
        )
        output.append(dict(run=run_name, task="correct", metric="balanced accuracy change (visible - standard)", n=len(sub), recordings=len({k[0] for k in sub}), value=change[0], ci_low=change[1], ci_high=change[2]))

        multi = [k for k in keys if k[2] == "error_types" and k in data]
        add(run_name, "exact match", multi, lambda k, r: r["correct"], kind="error_types")
        add(run_name, "exact match without F/G", multi, lambda k, r: exact_match_without_fg(r["predicted"], r["target"]), kind="error_types")
        add(run_name, "answers empty list", multi, lambda k, r: as_labels(r["predicted"]) == [], kind="error_types")

    for comparator in ("k8", "kall"):
        if comparator not in runs or "oracle" not in runs:
            continue
        for label in ("standard", "visible label"):
            paired = [k for k in correct_keys if k in runs["oracle"] and k in runs[comparator]]

            def paired_balanced(sample, label=label, comparator=comparator):
                def target(k):
                    truth = gt[k[0], k[1]]
                    return truth["target"] if label == "standard" else visible_target(truth["target"], truth["group"])

                pos = [k for k in sample if target(k)]
                neg = [k for k in sample if not target(k)]
                if not pos or not neg:
                    return float("nan")
                scores = []
                for name in ("oracle", comparator):
                    data = runs[name]
                    positive_accuracy = _rate(data, pos, lambda k, r: bool_is_correct(r["predicted"], target(k)))
                    negative_accuracy = _rate(data, neg, lambda k, r: bool_is_correct(r["predicted"], target(k)))
                    scores.append(50 * (positive_accuracy + negative_accuracy))
                return scores[0] - scores[1]

            if paired:
                val = percentile_ci(paired_balanced, paired, rng)
                output.append(dict(run=f"oracle - {comparator}", task="paired", metric=f"balanced accuracy difference ({label})", n=len(paired), recordings=len({k[0] for k in paired}), value=val[0], ci_low=val[1], ci_high=val[2]))
    return output


def validate(rows: list[dict], group_counts: dict[str, int], run_names: list[str]) -> None:
    if group_counts != EXPECTED_GROUPS:
        raise ValueError(f"GT group counts mismatch: expected {EXPECTED_GROUPS}, got {group_counts}")
    if len(run_names) != 3:
        raise ValueError("expected exactly three RUN arguments: oracle, k=8, k=all")
    roles = ("oracle", "k8", "kall")
    for role, run in zip(roles, run_names):
        values = {r["metric"]: r["value"] for r in rows if r["run"] == run}
        names = ("accuracy (standard)", "says correct (standard)", "accuracy on steps labeled correct (standard)", "accuracy on steps labeled wrong (standard)", "balanced accuracy (standard)")
        if any(n not in values for n in names):
            raise ValueError(f"missing validation metrics for {role}/{run}: {values}")
        actual = tuple(values[n] for n in names)
        expected = EXPECTED[role]
        if any(abs(a - e) > 0.1 for a, e in zip(actual, expected)):
            raise ValueError(f"documented-number validation failed for {role}/{run}: expected {expected}, got {actual}")


def print_markdown(rows: list[dict], run_labels: dict[str, str]) -> None:
    print("| Run | Metric | n | Recordings | Estimate | 95% recording-bootstrap CI |")
    print("|---|---|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {run_labels[r['run']]} | {r['metric']} | {r['n']} | {r['recordings']} | {r['value']:.2f}% | [{r['ci_low']:.2f}, {r['ci_high']:.2f}]% |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if len(args.runs) != 3:
        parser.error("pass exactly three runs, ordered oracle k8 kall")
    source_paths = [newest_scores(args.runs_dir, run) for run in args.runs]
    loaded = {run: read_run(path, oracle=(i == 0)) for i, (run, path) in enumerate(zip(args.runs, source_paths))}
    data = {role: loaded[run] for role, run in zip(("oracle", "k8", "kall"), args.runs)}
    oracle_steps = {(recording, step) for recording, step, field in data["oracle"] if field == "correct"}
    gt, group_counts = read_ground_truth(
        Path("data/cc4d_dense_v1/ground_truth.jsonl"), oracle_steps
    )
    common = set.intersection(*(set(d) for d in data.values()))
    # Intersect by field: one run is missing one error_types row, but that must not
    # remove the corresponding stepNN_correct row from the boolean analysis.
    keys = sorted(common)
    if not keys:
        raise SystemExit("no intersecting performed steps with both fields across all runs")
    rows = analyze(data, gt, keys, np.random.default_rng(SEED))
    validate(rows, group_counts, ("oracle", "k8", "kall"))
    print_markdown(rows, {"oracle": "oracle", "k8": "k=8", "kall": "k=all", "oracle - k8": "oracle - k=8", "oracle - kall": "oracle - k=all"})
    print("\nSources:")
    for run, path in zip(args.runs, source_paths):
        print(f"- {run}: {path} (evaluation {path.parent.name})")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "metrics.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (args.out / "sources.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["role", "run", "field_scores", "evaluation_id"])
            writer.writerows([[role, run, str(path), path.parent.name] for role, run, path in zip(("oracle", "k8", "kall"), args.runs, source_paths)])
    print(f"\nBootstrap: {BOOTSTRAPS} recording resamples; seed {SEED}.")


if __name__ == "__main__":
    main()
