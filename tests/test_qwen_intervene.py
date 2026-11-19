import importlib
from pathlib import Path

import pytest


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("qwen_intervene")


def test_parse_conditions(mod):
    got = mod.parse_conditions("base,a1.5,all2,text3")
    assert [(c["name"], c["alpha"], c["layers"], c["target"]) for c in got] == [
        ("base", 1.0, "late", mod.VIDEO),
        ("a1.5", 1.5, "late", mod.VIDEO),
        ("all2", 2.0, "all", mod.VIDEO),
        ("text3", 3.0, "late", mod.TEXT),
    ]


@pytest.mark.parametrize("bad", ["a", "x2", "a0", "a-1"])
def test_parse_conditions_rejects(mod, bad):
    with pytest.raises(SystemExit):
        mod.parse_conditions(bad)
