"""Aggregate field scores, execution status, usage and cost into run-level metrics.

Conventions:
* Accuracy denominators are ALL selected fields / reports. Missing, invalid and not-executed
  answers count as incorrect; nothing is dropped.
* Metrics are ``provisional`` whenever any call has not succeeded or a writer is active.
* Usage/cost: ``None`` = unknown, never zero. Totals carry a status:
  ``complete`` | ``lower_bound`` (some attempts may have been billed without usage) | ``unknown``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from video_report.config import PricingConfig
from video_report.evaluation.scorers import VALID_ANSWER_OUTCOMES, FieldScore

# Attempt outcomes whose request may have been processed (and billed) with no usage recorded.
_POSSIBLY_BILLED = frozenset({"timeout", "interrupted"})


def _rate(num: int, den: int) -> dict[str, Any]:
    return {"value": (num / den) if den else None, "numerator": num, "denominator": den}


def _attempt_usage_class(attempt: dict[str, Any]) -> str:
    """known | success_no_usage | possibly_billed | assumed_unbilled | in_flight"""
    outcome = attempt["outcome"]
    if outcome == "success":
        u = attempt.get("usage")
        if u and u.get("input_tokens") is not None and u.get("output_tokens") is not None:
            return "known"
        return "success_no_usage"
    if outcome in _POSSIBLY_BILLED:
        return "possibly_billed"
    if outcome == "in_progress":
        return "in_flight"
    err = attempt.get("error") or {}
    return "possibly_billed" if err.get("outcome_unknown") else "assumed_unbilled"


def _attempt_cost(usage: dict[str, Any], pricing: PricingConfig) -> float | None:
    inp = usage["input_tokens"]
    out = usage["output_tokens"] + (usage.get("reasoning_tokens") or 0)
    cached = usage.get("cached_input_tokens") or 0
    if cached and pricing.cached_input_per_million is None:
        return None
    cached_price = pricing.cached_input_per_million or 0.0
    cost: float = (
        (inp - cached) * pricing.input_per_million
        + cached * cached_price
        + out * pricing.output_per_million
    ) / 1_000_000
    return cost


def _usage_block(
    attempts: Sequence[dict[str, Any]], pricing: PricingConfig | None
) -> dict[str, Any]:
    classes = Counter(_attempt_usage_class(a) for a in attempts)
    tokens = {k: 0 for k in ("input", "output", "reasoning", "cached_input", "total")}
    cost = 0.0
    cost_known = pricing is not None
    for a in attempts:
        if _attempt_usage_class(a) != "known":
            continue
        u = a["usage"]
        tokens["input"] += u["input_tokens"]
        tokens["output"] += u["output_tokens"]
        tokens["reasoning"] += u.get("reasoning_tokens") or 0
        tokens["cached_input"] += u.get("cached_input_tokens") or 0
        tokens["total"] += u.get("total_tokens") or 0
        if pricing is not None:
            c = _attempt_cost(u, pricing)
            if c is None:
                cost_known = False
            else:
                cost += c
    gaps = classes["success_no_usage"] + classes["possibly_billed"] + classes["in_flight"]
    if classes["known"] == 0:
        status = "unknown"
    elif gaps:
        status = "lower_bound"
    else:
        status = "complete"
    cost_status = status if cost_known else "unknown"
    return {
        "attempts": dict(classes),
        "tokens": tokens if classes["known"] else None,
        "tokens_status": status,
        "cost": round(cost, 8) if cost_status != "unknown" else None,
        "cost_status": cost_status,
    }


def aggregate(
    scores: Sequence[FieldScore],
    calls: Sequence[dict[str, Any]],
    pricing: PricingConfig | None,
    *,
    writer_active: bool,
) -> dict[str, Any]:
    n_fields = len(scores)
    outcomes = Counter(s.outcome for s in scores)
    n_correct = outcomes["correct"]
    n_valid = sum(1 for s in scores if s.outcome in VALID_ANSWER_OUTCOMES)

    by_report: dict[str, list[FieldScore]] = defaultdict(list)
    for s in scores:
        by_report[s.report_id].append(s)
    calls_by_report: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in calls:
        calls_by_report[c["report_id"]].append(c)

    report_rows = []
    for rid, fs in by_report.items():
        rcalls = calls_by_report[rid]
        executed = all(c["status"] == "succeeded" for c in rcalls)
        attempts = [a for c in rcalls for a in c["attempts"]]
        usage = _usage_block(attempts, pricing)
        report_rows.append(
            {
                "report_id": rid,
                "num_fields": len(fs),
                "correct": sum(s.correct for s in fs),
                "valid_answers": sum(s.outcome in VALID_ANSWER_OUTCOMES for s in fs),
                "complete": all(s.outcome in VALID_ANSWER_OUTCOMES for s in fs),
                "all_correct": all(s.correct for s in fs),
                "calls_executed": executed,
                "attempts": len(attempts),
                "cost": usage["cost"],
                "cost_status": usage["cost_status"],
                "tokens": usage["tokens"],
            }
        )
    n_reports = len(report_rows)

    call_status = Counter(c["status"] for c in calls)
    n_calls = len(calls)
    attempts_all = [a for c in calls for a in c["attempts"]]
    attempt_outcomes = Counter(a["outcome"] for a in attempts_all)
    failure_kinds = Counter(
        (a.get("error") or {}).get("kind", "unknown")
        for a in attempts_all
        if a["outcome"] not in ("success", "in_progress")
    )
    failed_calls = Counter(c.get("failure", "unknown") for c in calls if c["status"] == "failed")

    provisional_reasons = []
    not_done = n_calls - call_status["succeeded"]
    if not_done:
        provisional_reasons.append(
            f"{not_done} of {n_calls} calls have not succeeded "
            f"({dict((k, v) for k, v in call_status.items() if k != 'succeeded')}); their "
            f"fields are scored as not_executed (incorrect)"
        )
    if writer_active:
        provisional_reasons.append("a writer was active on the run during evaluation")

    warnings = []
    if n_calls and call_status["succeeded"] == 0:
        warnings.append(
            "no call succeeded: accuracy is 0 because nothing executed, which says nothing "
            "about model quality; cost is not reported as zero"
        )

    completed_costs = [
        r["cost"] for r in report_rows if r["calls_executed"] and r["cost_status"] == "complete"
    ]
    n_completed = sum(r["calls_executed"] for r in report_rows)
    run_usage = _usage_block(attempts_all, pricing)
    timeout_note = None
    if run_usage["attempts"].get("possibly_billed"):
        timeout_note = (
            "some attempts timed out or were interrupted after the request may have reached the "
            "provider; observed usage/cost is a lower bound"
        )

    return {
        "provisional": bool(provisional_reasons),
        "provisional_reasons": provisional_reasons,
        "warnings": warnings,
        "fields": {
            "field_accuracy": _rate(n_correct, n_fields),
            "valid_answer_rate": _rate(n_valid, n_fields),
            "outcomes": {
                o: outcomes.get(o, 0)
                for o in (
                    "correct",
                    "incorrect",
                    "invalid_answer",
                    "missing_answer",
                    "invalid_output",
                    "not_executed",
                )
            },
        },
        "reports": {
            "report_completeness_rate": _rate(sum(r["complete"] for r in report_rows), n_reports),
            "all_fields_correct_rate": _rate(sum(r["all_correct"] for r in report_rows), n_reports),
            "per_report": sorted(report_rows, key=lambda r: r["report_id"]),
        },
        "execution": {
            "execution_completion": _rate(call_status["succeeded"], n_calls),
            "calls_by_status": dict(call_status),
            "failed_calls_by_reason": dict(failed_calls),
            "attempts_total": len(attempts_all),
            "attempt_outcomes": dict(attempt_outcomes),
            "failed_attempt_kinds": dict(failure_kinds),
        },
        "usage": {
            **run_usage,
            "currency": pricing.currency if pricing else None,
            "pricing_source": pricing.source if pricing else None,
            "completed_reports": n_completed,
            "cost_per_completed_report": {
                "mean": (sum(completed_costs) / len(completed_costs)) if completed_costs else None,
                "reports_with_known_cost": len(completed_costs),
                "reports_with_unknown_cost": n_completed - len(completed_costs),
            },
            "note": timeout_note,
        },
    }
