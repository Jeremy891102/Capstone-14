import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "results" / "cc4d_oracle_order.py"
spec = importlib.util.spec_from_file_location("cc4d_oracle_order", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_step_groups_and_order_only_relabeling():
    assert mod.step_group(True, []) == "done correctly"
    assert mod.step_group(False, ["Order Error"]) == "Order only"
    assert mod.step_group(False, ["Order Error", "Technique Error"]) == "Order + other"
    assert mod.step_group(False, ["Timing Error"]) == "no Order"
    assert mod.visible_target(False, "Order only") is True
    assert mod.visible_target(False, "Order + other") is False


def test_strip_fg_and_exact_match():
    assert mod.strip_order_missing(["A", "F", "G"]) == ["A"]
    assert mod.strip_order_missing(["F", "G"]) == []
    assert mod.exact_match_without_fg(["A", "F"], ["A", "G"])
    assert not mod.exact_match_without_fg(["A"], ["B"])
