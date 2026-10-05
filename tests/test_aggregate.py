"""Aggregation checked against independently hand-calculated values."""

from __future__ import annotations

from typing import Any

import pytest

from video_report.config import PricingConfig
from video_report.evaluation.aggregate import aggregate
from video_report.evaluation.scorers import FieldScore


def fs(report: str, field: str, outcome: str, call: str = "c") -> FieldScore:
    return FieldScore(
        report_id=report,
        field_id=field,
        call_id=call,
        outcome=outcome,  # type: ignore[arg-type]
        correct=outcome == "correct",
        predicted=None,
        predicted_text=None,
        target="A",
        target_text="x",
    )


def call(report: str, status: str, attempts: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    return {
        "call_id": f"{report}-{status}",
        "report_id": report,
        "field_ids": [],
        "status": status,
        "attempts": attempts,
        **kw,
    }


def ok(inp: int | None = 100, out: int | None = 10) -> dict[str, Any]:
    usage = None if inp is None else {"input_tokens": inp, "output_tokens": out}
    return {"outcome": "success", "usage": usage}


PRICING = PricingConfig(input_per_million=1.0, output_per_million=10.0, source="test fixture")


def test_hand_calculated_metrics() -> None:
    # Report r1 (3 fields): correct, correct, correct        -> complete, all correct
    # Report r2 (3 fields): correct, incorrect, invalid      -> incomplete
    # Report r3 (2 fields): incorrect, incorrect             -> complete, not all correct
    # Report r4 (2 fields): not_executed x2 (call failed)    -> incomplete
    # Fields: 10 total, 4 correct -> accuracy 4/10 = 0.4
    # Valid answers: r1 3 + r2 2 + r3 2 + r4 0 = 7 -> 7/10
    # Report completeness: r1, r3 = 2/4 = 0.5 ; all-fields-correct: r1 = 1/4 = 0.25
    scores = [
        fs("r1", "a", "correct"),
        fs("r1", "b", "correct"),
        fs("r1", "c", "correct"),
        fs("r2", "a", "correct"),
        fs("r2", "b", "incorrect"),
        fs("r2", "c", "invalid_answer"),
        fs("r3", "a", "incorrect"),
        fs("r3", "b", "incorrect"),
        fs("r4", "a", "not_executed"),
        fs("r4", "b", "not_executed"),
    ]
    calls = [
        call("r1", "succeeded", [ok(100, 10)]),
        call(
            "r2",
            "succeeded",
            [{"outcome": "retryable_error", "error": {"kind": "http_503"}}, ok(200, 20)],
        ),
        call("r3", "succeeded", [ok(300, 30)]),
        call(
            "r4",
            "failed",
            [{"outcome": "permanent_error", "error": {"kind": "http_400"}}],
            failure="permanent_error",
        ),
    ]
    m = aggregate(scores, calls, PRICING, writer_active=False)
    assert m["fields"]["field_accuracy"] == {"value": 0.4, "numerator": 4, "denominator": 10}
    assert m["fields"]["valid_answer_rate"]["value"] == pytest.approx(0.7)
    assert m["reports"]["report_completeness_rate"] == {
        "value": 0.5,
        "numerator": 2,
        "denominator": 4,
    }
    assert m["reports"]["all_fields_correct_rate"]["value"] == 0.25
    assert m["fields"]["outcomes"] == {
        "correct": 4,
        "incorrect": 3,
        "invalid_answer": 1,
        "missing_answer": 0,
        "invalid_output": 0,
        "not_executed": 2,
    }
    # Execution: 3 of 4 calls succeeded; failures reported separately from accuracy.
    assert m["execution"]["execution_completion"]["value"] == 0.75
    assert m["execution"]["failed_calls_by_reason"] == {"permanent_error": 1}
    assert m["execution"]["failed_attempt_kinds"] == {"http_503": 1, "http_400": 1}
    assert m["provisional"] is True
    # Usage: 600 in, 60 out -> 600*1/1e6 + 60*10/1e6 = 0.0006 + 0.0006 = 0.0012
    assert m["usage"]["tokens"]["input"] == 600 and m["usage"]["tokens"]["output"] == 60
    assert m["usage"]["cost"] == pytest.approx(0.0012)
    assert m["usage"]["cost_status"] == "complete"
    # Per completed report (r1, r2, r3): costs 0.0002, 0.0004 (incl. failed attempt, unbilled),
    # 0.0006 -> mean 0.0004
    assert m["usage"]["cost_per_completed_report"]["mean"] == pytest.approx(0.0004)
    assert m["usage"]["completed_reports"] == 3


def test_all_failed_run_is_not_reported_as_perfect_or_free() -> None:
    scores = [fs("r1", "a", "not_executed"), fs("r1", "b", "not_executed")]
    calls = [
        call(
            "r1",
            "failed",
            [{"outcome": "timeout", "error": {"kind": "timeout", "outcome_unknown": True}}] * 3,
            failure="retries_exhausted",
        )
    ]
    m = aggregate(scores, calls, PRICING, writer_active=False)
    assert m["fields"]["field_accuracy"] == {"value": 0.0, "numerator": 0, "denominator": 2}
    assert m["execution"]["execution_completion"]["value"] == 0.0
    assert m["provisional"] is True
    assert any("no call succeeded" in w for w in m["warnings"])
    assert m["usage"]["cost"] is None and m["usage"]["cost_status"] == "unknown"
    assert m["usage"]["tokens"] is None
    assert "lower bound" in m["usage"]["note"]
    assert m["usage"]["cost_per_completed_report"]["mean"] is None


def test_missing_usage_or_pricing_is_unknown_not_zero() -> None:
    scores = [fs("r1", "a", "correct")]
    m = aggregate(scores, [call("r1", "succeeded", [ok(None)])], PRICING, writer_active=False)
    assert m["usage"]["cost"] is None and m["usage"]["cost_status"] == "unknown"
    m = aggregate(scores, [call("r1", "succeeded", [ok()])], None, writer_active=False)
    assert m["usage"]["cost"] is None and m["usage"]["cost_status"] == "unknown"
    assert m["usage"]["tokens"]["input"] == 100  # tokens are still reported


def test_timeout_makes_cost_a_lower_bound() -> None:
    scores = [fs("r1", "a", "correct")]
    attempts = [{"outcome": "timeout", "error": {"kind": "timeout", "outcome_unknown": True}}, ok()]
    m = aggregate(scores, [call("r1", "succeeded", attempts)], PRICING, writer_active=False)
    assert m["usage"]["cost_status"] == "lower_bound"
    assert m["usage"]["cost"] == pytest.approx(0.0002)
    assert m["reports"]["per_report"][0]["cost_status"] == "lower_bound"
    assert m["usage"]["cost_per_completed_report"]["reports_with_unknown_cost"] == 1


def test_cached_tokens_without_cached_price_make_cost_unknown() -> None:
    a = {
        "outcome": "success",
        "usage": {"input_tokens": 100, "output_tokens": 1, "cached_input_tokens": 50},
    }
    m = aggregate(
        [fs("r1", "a", "correct")], [call("r1", "succeeded", [a])], PRICING, writer_active=False
    )
    assert m["usage"]["cost"] is None


def test_active_writer_marks_provisional() -> None:
    m = aggregate(
        [fs("r1", "a", "correct")], [call("r1", "succeeded", [ok()])], None, writer_active=True
    )
    assert m["provisional"] is True
