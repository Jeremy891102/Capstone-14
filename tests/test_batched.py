import pytest
from pydantic import ValidationError

from video_report.config import MethodConfig
from video_report.methods import ReportSelection, check_plan
from video_report.methods.batched import plan_batched


@pytest.mark.parametrize("size", [1, 4, 8, 16, 32])
def test_same_pool_order_and_video_boundaries(size):
    selections = [ReportSelection(r, tuple(f"q{i}" for i in range(32))) for r in ["a", "b"]]
    plan = plan_batched(selections, size)
    check_plan(plan, selections)
    assert len(plan) == 2 * 32 // size
    for s in selections:
        calls = [c for c in plan if c.report_id == s.report_id]
        assert all(len(c.field_ids) == size for c in calls)
        assert tuple(f for c in calls for f in c.field_ids) == s.field_ids


def test_partial_batch_does_not_drop_fields():
    s = [ReportSelection("a", ("x", "y", "z"))]
    plan = plan_batched(s, 2)
    check_plan(plan, s)
    assert [c.field_ids for c in plan] == [("x", "y"), ("z",)]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_batch_size(value):
    with pytest.raises(ValueError):
        plan_batched([], value)


def test_batch_config_requires_explicit_valid_size():
    assert MethodConfig(name="batched", batch_size=8).batch_size == 8
    for args in [{"name": "batched"}, {"name": "per_field", "batch_size": 8}]:
        with pytest.raises(ValidationError):
            MethodConfig(**args)
