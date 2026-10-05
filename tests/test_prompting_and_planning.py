"""Prompt construction, choice mapping, context whitelist, leakage, and call planning."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from video_report.benchmarks.schema import FieldSpec
from video_report.methods import METHODS, CallSpec, ReportSelection, check_plan
from video_report.prompting import PromptError, PromptTemplates, render
from video_report.snapshot import build_snapshot

REPO = Path(__file__).resolve().parents[1]
SNAPSHOTS = Path(__file__).parent / "snapshots"


# --------------------------------------------------------------------------- planning

SEL = [ReportSelection("r1", ("a", "b", "c")), ReportSelection("r2", ("a",))]


def test_whole_report_plan() -> None:
    plan = METHODS["whole_report"](SEL)
    assert plan == [
        CallSpec("r1::whole", "r1", ("a", "b", "c")),
        CallSpec("r2::whole", "r2", ("a",)),
    ]
    check_plan(plan, SEL)


def test_per_field_plan() -> None:
    plan = METHODS["per_field"](SEL)
    assert [c.call_id for c in plan] == [
        "r1::field=a",
        "r1::field=b",
        "r1::field=c",
        "r2::field=a",
    ]
    assert all(len(c.field_ids) == 1 for c in plan)
    check_plan(plan, SEL)


def test_plans_are_stable() -> None:
    for planner in METHODS.values():
        assert planner(SEL) == planner(list(SEL))


def test_check_plan_rejects_gaps_and_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate call ids"):
        check_plan([CallSpec("x", "r1", ("a",)), CallSpec("x", "r1", ("b", "c"))], SEL[:1])
    with pytest.raises(ValueError, match="exactly once"):
        check_plan([CallSpec("x", "r1", ("a", "b"))], SEL[:1])
    with pytest.raises(ValueError, match="exactly once"):
        check_plan([CallSpec("x", "r1", ("a", "b", "c")), CallSpec("y", "r1", ("a",))], SEL[:1])


# --------------------------------------------------------------------------- prompts

FIELDS = [
    FieldSpec(id="f1", question="Q one?", answer_type="single_choice", choices=["x", "y", "z"]),
    FieldSpec(id="f2", question="Q two?", answer_type="single_choice", choices=["p", "q"]),
]
T = PromptTemplates(None, "$context\n$questions\n$answer_format\n$field_ids $num_fields")


def test_choice_letters_preserve_order() -> None:
    _, user = render(T, FIELDS, {}, [])
    assert "A. x\nB. y\nC. z" in user and "A. p\nB. q" in user
    assert user.index("f1") < user.index("f2")
    fmt = json.loads(user[user.index("{") : user.rindex("}") + 1])
    assert list(fmt) == ["f1", "f2"]
    assert fmt["f1"] == "<one letter: A/B/C>"


def test_only_requested_fields_rendered() -> None:
    _, user = render(T, FIELDS[1:], {}, [])
    assert "f2" in user and "f1" not in user and "Q one?" not in user


def test_context_whitelist() -> None:
    ctx = {"task_name": "tea", "secret_hint": "DO_NOT_SEND"}
    _, user = render(T, FIELDS, ctx, ["task_name"])
    assert "task_name: tea" in user and "DO_NOT_SEND" not in user and "secret_hint" not in user
    _, user = render(T, FIELDS, ctx, [])
    assert "(none)" in user and "tea" not in user


@pytest.mark.parametrize(
    "user, msg",
    [
        ("$questions $target", "unknown placeholders"),
        ("$questions ${evidence}", "unknown placeholders"),
        ("no questions here", "must contain"),
        ("$questions costs $5", "invalid"),
    ],
)
def test_template_validation(user: str, msg: str) -> None:
    with pytest.raises(PromptError, match=msg):
        PromptTemplates(None, user).check()


def test_whole_report_prompt_snapshot() -> None:
    """Exact rendered prompt for the committed exp001 + synthetic report.

    To update intentionally: regenerate tests/snapshots/*.txt and review the diff.
    """
    snap = build_snapshot(REPO / "experiments/exp001_whole_mcq/config.yaml", environ={})
    req = snap.requests[0]
    assert req.system == (SNAPSHOTS / "exp001_system.txt").read_text()
    assert req.user == (SNAPSHOTS / "exp001_user.txt").read_text()


def test_per_field_prompt_snapshot() -> None:
    snap = build_snapshot(REPO / "experiments/exp002_per_field_mcq/config.yaml", environ={})
    assert [r.call_id for r in snap.requests] == [
        "synthetic-0001::field=primary_appliance",
        "synthetic-0001::field=first_item_in_cup",
        "synthetic-0001::field=task_outcome",
    ]
    assert snap.requests[1].user == (SNAPSHOTS / "exp002_first_item_in_cup_user.txt").read_text()


# --------------------------------------------------------------------------- leakage


@pytest.mark.parametrize("method", ["whole_report", "per_field"])
def test_targets_and_evidence_never_reach_requests(
    method: str, make_bench: Callable[..., Path], make_experiment: Callable[..., Path]
) -> None:
    bench = make_bench(
        {"r1": ["f1", "f2"], "r2": ["f1"]},
        targets={("r1", "f1"): "C", ("r1", "f2"): "B", ("r2", "f1"): "D"},
    )
    snap = build_snapshot(make_experiment(bench, method=method), environ={})
    blob = json.dumps([r.to_json() for r in snap.requests])
    for forbidden in (
        "EVIDENCE_SENTINEL",
        "METADATA_SENTINEL",
        "DO_NOT_SEND",
        "ann://",
        "ground_truth",
        '"target"',
        "synthetic",
    ):
        assert forbidden not in blob, forbidden
    # And the request objects have no target-bearing attribute at all.
    assert set(snap.requests[0].to_json()) == {
        "call_id",
        "report_id",
        "field_ids",
        "system",
        "user",
        "videos",
        "model",
        "generation",
        "fps",
    }


def test_committed_fixture_evidence_not_in_requests() -> None:
    for exp in ("exp001_whole_mcq", "exp002_per_field_mcq"):
        snap = build_snapshot(REPO / f"experiments/{exp}/config.yaml", environ={})
        blob = json.dumps([r.to_json() for r in snap.requests])
        assert "EVIDENCE_SENTINEL" not in blob and "recipe_step_count" not in blob
        assert "metadata is never sent" not in blob


def test_snapshot_works_without_ground_truth(
    make_bench: Callable[..., Path], make_experiment: Callable[..., Path]
) -> None:
    bench = make_bench({"r1": ["f1"]})
    (bench / "ground_truth.jsonl").unlink()
    assert len(build_snapshot(make_experiment(bench), environ={}).requests) == 1


def test_field_and_report_selection(
    make_bench: Callable[..., Path], make_experiment: Callable[..., Path]
) -> None:
    bench = make_bench({"r1": ["f1", "f2", "f3"], "r2": ["f1", "f2", "f3"]})
    cfg = make_experiment(
        bench,
        method="per_field",
        data={"benchmark": str(bench), "report_ids": ["r2"], "field_ids": ["f3", "f1"]},
    )
    snap = build_snapshot(cfg, environ={})
    assert [r.call_id for r in snap.requests] == ["r2::field=f1", "r2::field=f3"]
    assert [f.id for f in snap.reports[0].fields] == ["f1", "f3"]
